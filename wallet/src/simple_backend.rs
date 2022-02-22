#![deny(warnings)]
use anyhow::Result;
use ark_ed_on_bn254::Fq as Fr254;
use async_std::sync::RwLock;
use async_std::{channel, task};
use atomic_store::{
    load_store::BincodeLoadStore, AppendLog, AtomicStore, AtomicStoreLoader, PersistenceError,
    RollingLog,
};
use cap_rust_sandbox::{
    cape,
    deploy::EthMiddleware,
    ledger,
    ledger::CapeTransition,
    model::{CapeModelTxn, Erc20Code, EthereumAddr},
    types as sol,
    types::{CAPEEvents, CAPE},
};
use core::cmp;
use core::mem;
use core::time::Duration;
use ethers::abi::AbiDecode;
use ethers::prelude::*;
use futures::{select, FutureExt};
use jf_cap::{
    keys::{UserAddress, UserPubKey},
    structs::{Nullifier, ReceiverMemo, RecordCommitment, RecordOpening},
    TransactionNote,
};
use jf_primitives::merkle_tree::{FilledMTBuilder, MerkleFrontier, MerklePath, MerkleTree};
use net::server::{add_error_body, request_body, response};
use reef::traits::Transaction;
use serde::{Deserialize, Serialize};
use snafu::Snafu;
use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::path::Path;
use std::sync::Arc;
use tide::StatusCode;

#[derive(Clone, Debug, Snafu, Serialize, Deserialize)]
pub enum Error {
    #[snafu(display("Could not find relevant data: {}", msg))]
    NotFound { msg: String },

    #[snafu(display("failed to deserialize request body: {}", msg))]
    Deserialize { msg: String },

    #[snafu(display("submitted transaction does not form a valid block: {}", msg))]
    BadBlock { msg: String },

    #[snafu(display("error during transaction submission: {}", msg))]
    Submission { msg: String },

    #[snafu(display("transaction was not accepted by Ethereum miners"))]
    Rejected,

    #[snafu(display("internal server error: {}", msg))]
    Internal { msg: String },

    #[snafu(display("Started with wrong contract: expected {}, found {}", expected, found))]
    WrongContractConfig { expected: String, found: String },
}

impl net::Error for Error {
    fn catch_all(msg: String) -> Self {
        Self::Internal { msg }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NotFound,
            Self::Deserialize { .. } | Self::BadBlock { .. } => StatusCode::BadRequest,
            Self::Submission { .. }
            | Self::Rejected
            | Self::Internal { .. }
            | Self::WrongContractConfig { .. } => StatusCode::InternalServerError,
        }
    }
}

impl From<channel::SendError<CapeModelTxn>> for Error {
    fn from(err: channel::SendError<CapeModelTxn>) -> Self {
        Error::Internal {
            msg: format!("{:?}", err),
        }
    }
}

fn server_error<E: Into<Error>>(err: E) -> tide::Error {
    net::server_error(err)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemoInfo {
    pub memo: ReceiverMemo,
    pub rc: RecordCommitment,
    pub uid: u64,
    pub rc_merkle_path: MerklePath<Fr254>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemosEvent {
    pub memos: Vec<MemoInfo>,
    pub which_tx: Option<(u64, u64)>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct EtherStreamState {
    pub latest_block: u64,
    pub latest_tx_in_block: u64,
    pub pending_erc20_deposits: Vec<(Box<RecordOpening>, Erc20Code, EthereumAddr)>,
}

struct BackendStorage {
    store: AtomicStore,
    nul_log: AppendLog<BincodeLoadStore<Nullifier>>,
    nul_log_dirty: bool,
    address_log: AppendLog<BincodeLoadStore<UserPubKey>>,
    address_log_dirty: bool,
    // block, initial uid, frontier
    commit_log: AppendLog<BincodeLoadStore<(ledger::CapeBlock, u64, MerkleFrontier<Fr254>)>>, // from the start of time
    commit_log_dirty: bool,
    memo_log: AppendLog<BincodeLoadStore<MemosEvent>>,
    memo_log_dirty: bool,

    ether_stream_log: RollingLog<BincodeLoadStore<EtherStreamState>>,
    ether_stream_log_dirty: bool,

    contract_addr_log: RollingLog<BincodeLoadStore<(EthereumAddr, MerkleTree<Fr254>)>>,
    contract_addr_log_dirty: bool,
}

impl BackendStorage {
    #[allow(clippy::type_complexity)]
    fn open(
        storage_path: &Path,
    ) -> Result<(
        Self,
        Option<((EthereumAddr, MerkleTree<Fr254>), MerkleTree<Fr254>)>,
        HashSet<Nullifier>,
        HashMap<UserAddress, UserPubKey>,
    )> {
        let mut store_loader = AtomicStoreLoader::load(storage_path, "storage_runner_store")?;

        // todo: fixed size append log
        let nul_log = AppendLog::load(
            &mut store_loader,
            <BincodeLoadStore<_>>::default(),
            "nullifiers",
            1 << 12,
        )?;

        let address_log = AppendLog::load(
            &mut store_loader,
            <BincodeLoadStore<UserPubKey>>::default(),
            "addresses",
            1 << 12,
        )?;
        let commit_log = AppendLog::load(
            &mut store_loader,
            <BincodeLoadStore<(ledger::CapeBlock, u64, MerkleFrontier<Fr254>)>>::default(),
            "commits",
            1 << 12,
        )?;
        let memo_log = AppendLog::load(
            &mut store_loader,
            <BincodeLoadStore<MemosEvent>>::default(),
            "memos",
            1 << 12,
        )?;
        let ether_stream_log = RollingLog::load(
            &mut store_loader,
            <BincodeLoadStore<EtherStreamState>>::default(),
            "ether_stream",
            1 << 12,
        )?;
        let contract_addr_log = RollingLog::load(
            &mut store_loader,
            <BincodeLoadStore<(EthereumAddr, MerkleTree<Fr254>)>>::default(),
            "contract_addr",
            1 << 12,
        )?;

        let store = AtomicStore::open(store_loader)?;

        let nul_scan = std::thread::spawn(move || {
            let mut ret = HashSet::new();
            for n in nul_log.iter() {
                ret.insert(n.unwrap());
            }
            (ret, nul_log)
        });

        let address_scan = std::thread::spawn(move || {
            let mut ret = HashMap::new();
            for key in address_log.iter() {
                let key = key.unwrap();
                ret.insert(key.address(), key);
            }
            (ret, address_log)
        });

        let most_recent_addr = match contract_addr_log.load_latest() {
            Ok(x) => Ok(Some(x)),
            Err(PersistenceError::FailedToFindExpectedResource { .. }) => Ok(None),
            Err(e) => Err(e),
        }?;

        // TODO: crash unless this is the genesis block
        let merkle_tree = most_recent_addr.as_ref().map(|(addr, genesis)| {
            let mut builder = FilledMTBuilder::from_existing(genesis.clone()).unwrap();
            for comm in commit_log.iter() {
                let (comm, _, _) = comm.unwrap();
                for rc in comm
                    .0
                    .into_iter()
                    .flat_map(|tx| tx.output_commitments().into_iter())
                {
                    builder.push(rc.to_field_element());
                }
            }
            ((addr.clone(), genesis.clone()), builder.build())
        });

        // TODO: wrap these properly
        let (nuls, nul_log) = nul_scan.join().unwrap();
        let (addresses, address_log) = address_scan.join().unwrap();

        let ret = Self {
            store,
            nul_log,
            nul_log_dirty: false,
            address_log,
            address_log_dirty: false,
            commit_log,
            commit_log_dirty: false,
            memo_log,
            memo_log_dirty: false,
            ether_stream_log,
            ether_stream_log_dirty: false,
            contract_addr_log,
            contract_addr_log_dirty: false,
        };

        Ok((ret, merkle_tree, nuls, addresses))
    }

    fn commit(&mut self) {
        // TODO: should these be unwraps?
        // My guess is that crashing and burning is better than a "partial
        // commit" -- does atomic store handle that properly?
        if self.nul_log_dirty {
            self.nul_log_dirty = false;
            self.nul_log.commit_version().unwrap();
        } else {
            self.nul_log.skip_version().unwrap();
        }
        if self.address_log_dirty {
            self.address_log_dirty = false;
            self.address_log.commit_version().unwrap();
        } else {
            self.address_log.skip_version().unwrap();
        }
        if self.commit_log_dirty {
            self.commit_log_dirty = false;
            self.commit_log.commit_version().unwrap();
        } else {
            self.commit_log.skip_version().unwrap();
        }
        if self.memo_log_dirty {
            self.memo_log_dirty = false;
            self.memo_log.commit_version().unwrap();
        } else {
            self.memo_log.skip_version().unwrap();
        }
        if self.ether_stream_log_dirty {
            self.ether_stream_log_dirty = false;
            self.ether_stream_log.commit_version().unwrap();
        } else {
            self.ether_stream_log.skip_version().unwrap();
        }
        if self.contract_addr_log_dirty {
            self.contract_addr_log_dirty = false;
            self.contract_addr_log.commit_version().unwrap();
        } else {
            self.contract_addr_log.skip_version().unwrap();
        }

        self.store.commit_version().unwrap();
    }
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct WebState {
    rpc_client: Arc<EthMiddleware>,
    contract: CAPE<EthMiddleware>,
    storage: Arc<RwLock<BackendStorage>>,

    // TODO: something more clever for these?
    nullifiers: Arc<RwLock<HashSet<Nullifier>>>,
    addresses: Arc<RwLock<HashMap<UserAddress, UserPubKey>>>,
    merkle_tree: Arc<RwLock<MerkleTree<Fr254>>>,

    tx_queue: channel::Sender<CapeModelTxn>,

    // kill channel, task
    eqs_scan_task: Arc<RwLock<Option<(channel::Sender<()>, task::JoinHandle<()>)>>>,
    submit_task: Arc<RwLock<Option<(channel::Sender<()>, task::JoinHandle<()>)>>>,
}

const TX_PER_BLOCK: usize = 5;

async fn run_eqs_scan(
    eqs_kill_rx: channel::Receiver<()>,
    rpc_client: Arc<EthMiddleware>,
    contract: CAPE<EthMiddleware>,
    storage: Arc<RwLock<BackendStorage>>,
    nullifiers: Arc<RwLock<HashSet<Nullifier>>>,
    merkle_tree: Arc<RwLock<MerkleTree<Fr254>>>,
) {
    while eqs_kill_rx.try_recv().is_err() {
        let mut stream_state = {
            let storage = storage.read().await;
            storage
                .ether_stream_log
                .load_latest()
                .unwrap_or_else(|_| Default::default())
        };

        let mut upper_block_num = stream_state.latest_block + 0x100;
        if let Ok(current_block) = rpc_client.get_block_number().await {
            upper_block_num = cmp::min(current_block.as_u64(), upper_block_num);
        }
        stream_state.latest_block = cmp::min(stream_state.latest_block, upper_block_num);

        let query = contract
            .events()
            .from_block(stream_state.latest_block)
            .to_block(upper_block_num);

        for (event, meta) in query.query_with_meta().await.unwrap_or_default() {
            assert!(stream_state.latest_block <= meta.block_number.as_u64());
            if meta.block_number.as_u64() == stream_state.latest_block
                && meta.transaction_index.as_u64() <= stream_state.latest_tx_in_block
            {
                continue;
            }

            stream_state.latest_block = meta.block_number.as_u64();
            stream_state.latest_tx_in_block = meta.transaction_index.as_u64();

            match event {
                CAPEEvents::BlockCommittedFilter(committed) => {
                    // TODO: is this safe?
                    let eth_tx = rpc_client
                        .get_transaction(meta.transaction_hash)
                        .await
                        .unwrap()
                        .unwrap();

                    // if we can't find the data, something is wrong
                    // For example, someone called the cape contract
                    // through another contract
                    let decoded_calldata_block = contract
                        .decode::<sol::CapeBlock, _>("submitCapeBlock", eth_tx.input)
                        .unwrap();

                    let decoded_cape_block = cape::CapeBlock::from(decoded_calldata_block);
                    let (txns, _) = decoded_cape_block.to_cape_transactions().unwrap();

                    let deposits = mem::take(&mut stream_state.pending_erc20_deposits);

                    let transitions = txns
                        .into_iter()
                        .map(CapeTransition::Transaction)
                        .chain(deposits.into_iter().map(|(ro, erc20_code, src_addr)| {
                            CapeTransition::Wrap {
                                erc20_code,
                                src_addr,
                                ro,
                            }
                        }))
                        .collect::<Vec<_>>();

                    {
                        let mut storage = storage.write().await;
                        let mut nullifiers = nullifiers.write().await;
                        let mut merkle_tree = merkle_tree.write().await;

                        storage.nul_log_dirty = true;
                        storage.commit_log_dirty = true;
                        storage.ether_stream_log_dirty = true;

                        let final_state = stream_state.clone();

                        let (_, prev_height, _) = storage.commit_log.load_latest().unwrap();
                        assert_eq!(prev_height + 1, committed.height);

                        for tx in transitions.iter() {
                            for (n, _) in tx.proven_nullifiers() {
                                nullifiers.insert(n);
                                storage.nul_log.store_resource(&n).unwrap();
                            }
                            for rc in tx.output_commitments() {
                                merkle_tree.push(rc.into());
                            }
                        }

                        storage
                            .commit_log
                            .store_resource(&(
                                ledger::CapeBlock(transitions),
                                prev_height + 1,
                                merkle_tree.frontier(),
                            ))
                            .unwrap();
                        storage
                            .ether_stream_log
                            .store_resource(&final_state)
                            .unwrap();
                        storage.commit();
                    }
                }

                CAPEEvents::Erc20TokensDepositedFilter(deposited) => {
                    let ro_bytes = deposited.ro_bytes.clone();
                    let ro_sol: sol::RecordOpening = AbiDecode::decode(ro_bytes).unwrap();
                    let ro = RecordOpening::from(ro_sol);
                    stream_state.pending_erc20_deposits.push((
                        Box::new(ro),
                        Erc20Code::from(deposited.erc_20_token_address),
                        EthereumAddr::from(deposited.from),
                    ));

                    {
                        let mut storage = storage.write().await;
                        storage.ether_stream_log_dirty = true;

                        let final_state = stream_state.clone();
                        storage
                            .ether_stream_log
                            .store_resource(&final_state)
                            .unwrap();
                        storage.commit();
                    }
                }
            }
        }

        // TODO: slower?
        task::sleep(Duration::from_millis(500)).await;
    }
}

impl WebState {
    pub async fn new(
        storage_path: &Path,
        rpc_client: Arc<EthMiddleware>,
        beneficiary_key: UserAddress,
        contract_addr: EthereumAddr,
        genesis_records: MerkleTree<Fr254>,
        port: u64,
    ) -> Result<Self> {
        let (mut storage, most_recent_addr, nullifiers, addresses) =
            BackendStorage::open(storage_path)?;

        let genesis_info = (contract_addr.clone(), genesis_records.clone());

        let merkle_tree = match most_recent_addr {
            None => {
                storage
                    .contract_addr_log
                    .store_resource(&genesis_info)
                    .unwrap();
                storage.contract_addr_log_dirty = true;
                storage.commit();
                genesis_records
            }
            Some((other_gen_info, _)) if other_gen_info != genesis_info => {
                return Err(Error::WrongContractConfig {
                    expected: format!("{:?}", genesis_info),
                    found: format!("{:?}", other_gen_info),
                }
                .into());
            }
            Some((_other_gen_info, tree)) => tree,
        };

        let storage = Arc::new(RwLock::new(storage));
        let nullifiers = Arc::new(RwLock::new(nullifiers));
        let merkle_tree = Arc::new(RwLock::new(merkle_tree));
        let addresses = Arc::new(RwLock::new(addresses));

        let contract = CAPE::new(contract_addr, rpc_client.clone());

        let (tx_queue, tx_queue_rx) = channel::bounded::<CapeModelTxn>(10);
        let (eqs_kill, eqs_kill_rx) = channel::bounded::<()>(1);
        let (submit_kill, submit_kill_rx) = channel::bounded::<()>(1);

        let eqs_scan = task::spawn({
            let rpc_client = rpc_client.clone();
            let contract = contract.clone();
            let storage = storage.clone();
            let nullifiers = nullifiers.clone();
            let merkle_tree = merkle_tree.clone();
            async move {
                run_eqs_scan(
                    eqs_kill_rx,
                    rpc_client,
                    contract,
                    storage,
                    nullifiers,
                    merkle_tree,
                )
                .await
            }
        });

        let submit_task = task::spawn({
            let contract = contract.clone();
            async move {
                let mut timer = task::spawn(task::sleep(Duration::from_secs(15))).fuse();
                let mut txs_pending = vec![];
                loop {
                    let submit = select! {
                        _ = submit_kill_rx.recv().fuse() => { return; },
                        _ = timer => true,
                        tx = tx_queue_rx.recv().fuse() => {
                            // TODO: validation?
                            txs_pending.push(tx.unwrap());

                            txs_pending.len() >= TX_PER_BLOCK
                        },
                    };

                    if submit {
                        let txs = mem::take(&mut txs_pending);

                        if let Ok(cape_block) =
                            cape::CapeBlock::from_cape_transactions(txs, beneficiary_key.clone())
                        {
                            if let Ok(res) =
                                contract.submit_cape_block(cape_block.into()).send().await
                            {
                                let _ = res.await;
                            }
                        }

                        timer = task::spawn(task::sleep(Duration::from_secs(15))).fuse();
                    }
                }
            }
        });

        let ret = Self {
            rpc_client,
            merkle_tree,
            contract,
            storage,
            nullifiers,
            addresses,
            tx_queue,
            eqs_scan_task: Arc::new(RwLock::new(Some((eqs_kill, eqs_scan)))),
            submit_task: Arc::new(RwLock::new(Some((submit_kill, submit_task)))),
        };

        init_web_server(ret.clone(), port.to_string());
        wait_for_server(port).await;
        Ok(ret)
    }

    pub async fn shutdown(self) {
        if let Some((eqs_kill, eqs_task)) =
            mem::take(&mut self.eqs_scan_task.write().await.as_mut())
        {
            eqs_kill.send(()).await.unwrap();
            eqs_task.await;
        }

        if let Some((submit_kill, submit_task)) =
            mem::take(&mut self.eqs_scan_task.write().await.as_mut())
        {
            submit_kill.send(()).await.unwrap();
            submit_task.await;
        }
    }
}

async fn submit_endpoint(mut req: tide::Request<WebState>) -> Result<tide::Response, tide::Error> {
    let tx = request_body(&mut req).await.map_err(|err| {
        server_error(Error::Deserialize {
            msg: err.to_string(),
        })
    })?;
    let ret = &req.state().tx_queue.send(tx).await.map_err(server_error)?;
    response(&req, ret)
}

async fn get_public_key_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let addr: UserAddress = request_body(&mut req).await.map_err(|err| {
        server_error(Error::Deserialize {
            msg: err.to_string(),
        })
    })?;
    match req.state().addresses.read().await.get(&addr) {
        None => Ok(tide::Response::new(StatusCode::NotFound)),
        Some(x) => response(&req, x),
    }
}

async fn register_user_key_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let insert_request: address_book::InsertPubKey = net::server::request_body(&mut req).await?;
    let pub_key = address_book::verify_sig_and_get_pub_key(insert_request)?;

    let mut storage = req.state().storage.write().await;
    storage.address_log_dirty = true;
    storage.address_log.store_resource(&pub_key).unwrap();
    storage.commit();

    Ok(tide::Response::new(StatusCode::Ok))
}

async fn get_commit_event_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let ix: u64 = net::server::request_body(&mut req).await?;

    let storage = req.state().storage.read().await;
    let (blk, _, _) = storage
        .commit_log
        .iter()
        .nth(ix.try_into()?)
        .ok_or_else(|| Error::NotFound {
            msg: "out of bounds".to_string(),
        })??;
    response(&req, (blk, ix, ix))
}

async fn get_memo_bb_event_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let ix: u64 = net::server::request_body(&mut req).await?;

    let storage = req.state().storage.read().await;
    let memos = storage
        .memo_log
        .iter()
        .nth(ix.try_into()?)
        .ok_or_else(|| Error::NotFound {
            msg: "out of bounds".to_string(),
        })??;
    response(&req, memos)
}

async fn post_memos_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let (block_id, tx_id, memos, sig): (u64, u64, Vec<ReceiverMemo>, jf_cap::Signature) =
        net::server::request_body(&mut req).await?;

    let state = req.state();
    let mut storage = state.storage.write().await;
    let merkle_tree = state.merkle_tree.read().await;
    let (comm, mut tx_ix_offset, _) = storage
        .commit_log
        .iter()
        .nth(block_id.try_into()?)
        .ok_or_else(|| Error::NotFound {
            msg: "out of bounds".to_string(),
        })??;
    let tx_id = tx_id.try_into()?;
    let note = match comm.0.get(tx_id) {
        Some(CapeTransition::Transaction(CapeModelTxn::CAP(note))) => Ok(note.clone()),
        Some(CapeTransition::Transaction(CapeModelTxn::Burn { xfr, .. })) => {
            Ok(TransactionNote::Transfer(xfr.clone()))
        }
        // TODO: better error message
        _ => Err(Error::NotFound {
            msg: "Bad index when posting memos".to_string(),
        }),
    }?;

    for tx in comm.0[..tx_id].iter() {
        tx_ix_offset += tx.output_commitments().len() as u64;
    }

    note.verify_receiver_memos_signature(&memos, &sig)?;

    let memos = (tx_ix_offset..)
        .zip(memos.into_iter())
        .map(|(uid, memo)| {
            let (_, rc_merkle_leaf) = merkle_tree.get_leaf(uid).expect_ok().unwrap();
            MemoInfo {
                memo,
                rc: RecordCommitment::from_field_element(rc_merkle_leaf.leaf.0),
                uid,
                rc_merkle_path: rc_merkle_leaf.path,
            }
        })
        .collect();

    storage.memo_log_dirty = true;
    storage
        .memo_log
        .store_resource(&MemosEvent {
            memos,
            which_tx: Some((block_id, tx_id as u64)),
        })
        .unwrap();
    storage.commit();

    Ok(tide::Response::new(StatusCode::Ok))
}

async fn get_nullifier_proof_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let n: Nullifier = net::server::request_body(&mut req).await?;

    response(&req, req.state().nullifiers.read().await.contains(&n))
}

async fn get_current_frontier_endpoint(
    req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    response(&req, req.state().merkle_tree.read().await.frontier())
}

async fn get_transaction_endpoint(
    mut req: tide::Request<WebState>,
) -> Result<tide::Response, tide::Error> {
    let (block_id, tx_id): (u64, u64) = net::server::request_body(&mut req).await?;

    let state = req.state();
    let storage = state.storage.read().await;
    let (comm, _, _) = storage
        .commit_log
        .iter()
        .nth(block_id.try_into()?)
        .ok_or_else(|| Error::NotFound {
            msg: "out of bounds".to_string(),
        })??;
    response(
        &req,
        comm.0.get(usize::try_from(tx_id)?).ok_or_else(||
        // TODO: better error message
        Error::NotFound { msg: "Bad index for tx".to_string() })?,
    )
}

pub const DEFAULT_BACKEND_PORT: u16 = 50077u16;
pub const BACKEND_STARTUP_RETRIES: usize = 8;

pub fn init_web_server(
    contract: WebState,
    port: String,
) -> task::JoinHandle<Result<(), std::io::Error>> {
    let mut web_server = tide::with_state(contract);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/submit")
        .post(submit_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/get_public_key")
        .get(get_public_key_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/register_user_key")
        .post(register_user_key_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/get_commit_event")
        .get(get_commit_event_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/get_memo_bb_event")
        .get(get_memo_bb_event_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/post_memos")
        .post(post_memos_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/get_nullifier_proof")
        .get(get_nullifier_proof_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/get_current_frontier")
        .get(get_current_frontier_endpoint);
    web_server
        .with(add_error_body::<_, Error>)
        .at("/get_transaction")
        .get(get_transaction_endpoint);
    let addr = format!("0.0.0.0:{}", port);
    async_std::task::spawn(web_server.listen(addr))
}

pub async fn wait_for_server(port: u64) {
    // Wait for the server to come up and start serving.
    let mut backoff = Duration::from_millis(100);
    for _ in 0..BACKEND_STARTUP_RETRIES {
        if surf::connect(format!("http://localhost:{}", port))
            .send()
            .await
            .is_ok()
        {
            return;
        }
        task::sleep(backoff).await;
        backoff *= 2;
    }
    panic!("Simple backend did not start in {:?}", backoff);
}
