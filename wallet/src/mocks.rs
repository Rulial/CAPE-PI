// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Configurable Asset Privacy for Ethereum (CAPE) library.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Test-only implementation of the [reef] ledger abstraction for CAPE.

use crate::wallet::{CapeWalletBackend, CapeWalletError};
use async_std::{
    sync::{Mutex, MutexGuard},
    task::sleep,
};
use async_trait::async_trait;
use cap_rust_sandbox::{
    deploy::EthMiddleware, ledger::*, model::*, universal_param::UNIVERSAL_PARAM,
};
use commit::Committable;
use futures::stream::{iter, pending, Stream, StreamExt};
use itertools::izip;
use jf_cap::{
    keys::{UserAddress, UserKeyPair, UserPubKey},
    proof::{freeze::FreezeProvingKey, transfer::TransferProvingKey, UniversalParam},
    structs::{
        AssetCode, AssetDefinition, Nullifier, ReceiverMemo, RecordCommitment, RecordOpening,
    },
    KeyPair, MerklePath, MerkleTree, Signature, TransactionNote, VerKey,
};
use key_set::{OrderByOutputs, ProverKeySet, SizedKey, VerifierKeySet};
use rand_chacha::{rand_core::SeedableRng, ChaChaRng};
use reef::{
    traits::{Block as _, Transaction as _},
    Block,
};
use seahorse::{
    events::{EventIndex, EventSource, LedgerEvent},
    hd::KeyTree,
    loader::WalletLoader,
    persistence::AtomicWalletStorage,
    testing,
    txn_builder::{RecordDatabase, TransactionInfo, TransactionState, TransactionUID},
    WalletBackend, WalletError, WalletState,
};
use serde::{de::DeserializeOwned, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempdir::TempDir;
use testing::{MockEventSource, MockLedger, MockNetwork, SystemUnderTest};

pub fn test_asset_signing_key() -> KeyPair {
    KeyPair::generate(&mut ChaChaRng::from_seed([3; 32]))
}

#[derive(Clone)]
struct CommittedTransaction {
    txn: CapeTransition,
    uids: Vec<u64>,
    #[allow(clippy::type_complexity)]
    memos: Option<(
        Vec<(ReceiverMemo, RecordCommitment, u64, MerklePath)>,
        Signature,
    )>,
}

// A mock implementation of a CAPE network which maintains the full state of a CAPE ledger locally.
#[derive(Clone)]
pub struct MockCapeNetwork {
    contract: CapeContractState,
    call_data: HashMap<TransactionUID<CapeLedger>, (Vec<ReceiverMemo>, Signature)>,

    // Mock EQS and peripheral services
    block_height: u64,
    records: MerkleTree,
    // When an ERC20 deposit is finalized during a block submission, the contract emits an event
    // containing only the commitment of the new record. Therefore, to correlate these events with
    // the other information needed to reconstruct a CapeTransition::Wrap, the query service needs
    // to monitor the contracts Erc20Deposited events and keep track of the deposits which are
    // pending finalization.
    pending_erc20_deposits:
        HashMap<RecordCommitment, (Erc20Code, EthereumAddr, Box<RecordOpening>)>,
    events: MockEventSource<CapeLedger>,
    txns: HashMap<(u64, u64), CommittedTransaction>,
    address_map: HashMap<UserAddress, UserPubKey>,
}

impl MockCapeNetwork {
    pub fn new(
        verif_crs: VerifierKeySet,
        records: MerkleTree,
        initial_grant_memos: Vec<(ReceiverMemo, u64)>,
    ) -> Self {
        let mut ledger = Self {
            contract: CapeContractState::new(verif_crs, records.clone()),
            call_data: Default::default(),
            block_height: 0,
            records,
            pending_erc20_deposits: Default::default(),
            events: MockEventSource::new(EventSource::QueryService),
            txns: Default::default(),
            address_map: Default::default(),
        };

        // Broadcast receiver memos for the records which are included in the tree from the start,
        // so that clients can access records they have been granted at ledger setup time in a
        // uniform way.
        let memo_outputs = initial_grant_memos
            .into_iter()
            .map(|(memo, uid)| {
                let (comm, merkle_path) = ledger
                    .records
                    .get_leaf(uid)
                    .expect_ok()
                    .map(|(_, proof)| {
                        (
                            RecordCommitment::from_field_element(proof.leaf.0),
                            proof.path,
                        )
                    })
                    .unwrap();
                (memo, comm, uid, merkle_path)
            })
            .collect();
        ledger.generate_event(LedgerEvent::Memos {
            outputs: memo_outputs,
            transaction: None,
        });

        ledger
    }

    pub fn register_erc20(
        &mut self,
        asset_def: AssetDefinition,
        erc20_code: Erc20Code,
        sponsor_addr: EthereumAddr,
    ) -> Result<(), CapeValidationError> {
        self.submit_operations(vec![CapeModelOperation::RegisterErc20 {
            asset_def: Box::new(asset_def),
            erc20_code,
            sponsor_addr,
        }])
    }

    pub fn wrap_erc20(
        &mut self,
        erc20_code: Erc20Code,
        src_addr: EthereumAddr,
        ro: RecordOpening,
    ) -> Result<(), CapeValidationError> {
        self.submit_operations(vec![CapeModelOperation::WrapErc20 {
            erc20_code,
            src_addr,
            ro: Box::new(ro),
        }])
    }

    pub fn create_wallet<'a>(
        &self,
        univ_param: &'a UniversalParam,
    ) -> Result<WalletState<'a, CapeLedger>, CapeWalletError> {
        // Construct proving keys of the same arities as the verifier keys from the validator.
        let proving_keys = Arc::new(ProverKeySet {
            mint: jf_cap::proof::mint::preprocess(univ_param, CAPE_MERKLE_HEIGHT)
                .map_err(|source| CapeWalletError::CryptoError { source })?
                .0,
            freeze: self
                .contract
                .verif_crs
                .freeze
                .iter()
                .map(|k| {
                    Ok::<FreezeProvingKey, WalletError<CapeLedger>>(
                        jf_cap::proof::freeze::preprocess(
                            univ_param,
                            k.num_inputs(),
                            CAPE_MERKLE_HEIGHT,
                        )
                        .map_err(|source| CapeWalletError::CryptoError { source })?
                        .0,
                    )
                })
                .collect::<Result<_, _>>()?,
            xfr: self
                .contract
                .verif_crs
                .xfr
                .iter()
                .map(|k| {
                    Ok::<TransferProvingKey, WalletError<CapeLedger>>(
                        jf_cap::proof::transfer::preprocess(
                            univ_param,
                            k.num_inputs(),
                            k.num_outputs(),
                            CAPE_MERKLE_HEIGHT,
                        )
                        .map_err(|source| CapeWalletError::CryptoError { source })?
                        .0,
                    )
                })
                .collect::<Result<_, _>>()?,
        });

        // `records` should be _almost_ completely sparse. However, even a fully pruned Merkle tree
        // contains the last leaf appended, but as a new wallet, we don't care about _any_ of the
        // leaves, so make a note to forget the last one once more leaves have been appended.
        let record_mt = self.records.clone();
        let merkle_leaf_to_forget = if record_mt.num_leaves() > 0 {
            Some(record_mt.num_leaves() - 1)
        } else {
            None
        };

        Ok(WalletState {
            proving_keys,
            txn_state: TransactionState {
                validator: CapeTruster::new(self.block_height, record_mt.num_leaves()),
                now: self.now(),
                nullifiers: Default::default(),
                // Completely sparse nullifier set
                record_mt,
                records: RecordDatabase::default(),
                merkle_leaf_to_forget,

                transactions: Default::default(),
            },
            key_state: Default::default(),
            assets: Default::default(),
            viewing_accounts: Default::default(),
            freezing_accounts: Default::default(),
            sending_accounts: Default::default(),
        })
    }

    pub fn subscribe(
        &mut self,
        from: EventIndex,
        to: Option<EventIndex>,
    ) -> Pin<Box<dyn Stream<Item = (LedgerEvent<CapeLedger>, EventSource)> + Send>> {
        self.events.subscribe(from, to)
    }

    pub fn get_public_key(&self, address: &UserAddress) -> Result<UserPubKey, CapeWalletError> {
        Ok(self
            .address_map
            .get(address)
            .ok_or(CapeWalletError::Failed {
                msg: String::from("invalid user address"),
            })?
            .clone())
    }

    pub fn nullifier_spent(&self, nullifier: Nullifier) -> bool {
        self.contract.nullifiers.contains(&nullifier)
    }

    pub fn get_transaction(
        &self,
        block_id: u64,
        txn_id: u64,
    ) -> Result<CapeTransition, CapeWalletError> {
        Ok(self
            .txns
            .get(&(block_id, txn_id))
            .ok_or(CapeWalletError::Failed {
                msg: String::from("invalid transaction ID"),
            })?
            .txn
            .clone())
    }

    pub fn register_user_key(&mut self, key_pair: &UserKeyPair) -> Result<(), CapeWalletError> {
        let pub_key = key_pair.pub_key();
        self.address_map.insert(pub_key.address(), pub_key);
        Ok(())
    }

    pub fn get_wrapped_asset(
        &self,
        asset: &AssetCode,
    ) -> Result<Option<Erc20Code>, CapeWalletError> {
        match self
            .contract
            .erc20_registrar
            .iter()
            .find(|(definition, _)| definition.code == *asset)
        {
            Some((_, (erc20_code, _))) => Ok(Some(erc20_code.clone())),
            None => Ok(None),
        }
    }

    pub fn store_call_data(
        &mut self,
        txn: TransactionUID<CapeLedger>,
        memos: Vec<ReceiverMemo>,
        sig: Signature,
    ) {
        self.call_data.insert(txn, (memos, sig));
    }

    pub fn submit_operations(
        &mut self,
        ops: Vec<CapeModelOperation>,
    ) -> Result<(), CapeValidationError> {
        let (new_state, effects) = self.contract.submit_operations(ops)?;
        let mut events = vec![];
        for effect in effects {
            if let CapeModelEthEffect::Emit(event) = effect {
                events.push(event);
            }
        }

        // Simulate the EQS processing the events emitted by the contract, updating its state, and
        // broadcasting processed events to subscribers.
        for event in events {
            self.handle_event(event);
        }
        self.contract = new_state;

        Ok(())
    }

    fn handle_event(&mut self, event: CapeModelEvent) {
        match event {
            CapeModelEvent::BlockCommitted { txns, wraps } => {
                // Convert the transactions and wraps into CapeTransitions, and collect them all
                // into a single block, in the order they were processed by the contract
                // (transactions first, then wraps).
                let block = txns
                    .into_iter()
                    .map(CapeTransition::Transaction)
                    .chain(wraps.into_iter().map(|comm| {
                        // Look up the auxiliary information associated with this deposit which
                        // we saved when we processed the deposit event. This lookup cannot
                        // fail, because the contract only finalizes a Wrap operation after it
                        // has already processed the deposit, which involves emitting an
                        // Erc20Deposited event.
                        let (erc20_code, src_addr, ro) =
                            self.pending_erc20_deposits.remove(&comm).unwrap();
                        CapeTransition::Wrap {
                            erc20_code,
                            src_addr,
                            ro,
                        }
                    }))
                    .collect::<Vec<_>>();

                // Add transactions and outputs to query service data structures.
                for (i, txn) in block.iter().enumerate() {
                    let mut uids = Vec::new();
                    for comm in txn.output_commitments() {
                        uids.push(self.records.num_leaves());
                        self.records.push(comm.to_field_element());
                    }
                    self.txns.insert(
                        (self.block_height, i as u64),
                        CommittedTransaction {
                            txn: txn.clone(),
                            uids,
                            memos: None,
                        },
                    );
                }

                self.generate_event(LedgerEvent::Commit {
                    block: CapeBlock::new(block.clone()),
                    block_id: self.block_height,
                    state_comm: self.block_height + 1,
                });

                // The memos for this block should have already been posted in the calldata, so we
                // can now generate the corresponding Memos events.
                for (txn_id, txn) in block.into_iter().enumerate() {
                    if let Some((memos, sig)) = self.call_data.remove(&TransactionUID(txn.commit()))
                    {
                        self.post_memos(self.block_height, txn_id as u64, memos, sig)
                            .unwrap();
                    }
                }

                self.block_height += 1;
            }

            CapeModelEvent::Erc20Deposited {
                erc20_code,
                src_addr,
                ro,
            } => {
                self.pending_erc20_deposits
                    .insert(RecordCommitment::from(&*ro), (erc20_code, src_addr, ro));
            }
        }
    }
}

impl<'a> MockNetwork<'a, CapeLedger> for MockCapeNetwork {
    fn now(&self) -> EventIndex {
        self.events.now()
    }

    fn submit(&mut self, block: Block<CapeLedger>) -> Result<(), WalletError<CapeLedger>> {
        // Convert the submitted transactions to CapeOperations.
        let ops = block
            .txns()
            .into_iter()
            .map(|txn| match txn {
                CapeTransition::Transaction(txn) => CapeModelOperation::SubmitBlock(vec![txn]),
                CapeTransition::Wrap {
                    erc20_code,
                    src_addr,
                    ro,
                } => CapeModelOperation::WrapErc20 {
                    erc20_code,
                    src_addr,
                    ro,
                },
                CapeTransition::Faucet { .. } => {
                    panic!("submitting a Faucet transaction from a wallet is not supported")
                }
            })
            .collect();

        self.submit_operations(ops).map_err(cape_to_wallet_err)
    }

    fn post_memos(
        &mut self,
        block_id: u64,
        txn_id: u64,
        memos: Vec<ReceiverMemo>,
        sig: Signature,
    ) -> Result<(), WalletError<CapeLedger>> {
        let txn = match self.txns.get_mut(&(block_id, txn_id)) {
            Some(txn) => txn,
            None => {
                return Err(CapeWalletError::Failed {
                    msg: String::from("invalid transaction ID"),
                });
            }
        };
        if txn.memos.is_some() {
            return Err(CapeWalletError::Failed {
                msg: String::from("memos already posted"),
            });
        }

        // Validate the new memos.
        match &txn.txn {
            CapeTransition::Transaction(CapeModelTxn::CAP(note)) => {
                if note.verify_receiver_memos_signature(&memos, &sig).is_err() {
                    return Err(CapeWalletError::Failed {
                        msg: String::from("invalid memos signature"),
                    });
                }
                if memos.len() != txn.txn.output_len() {
                    return Err(CapeWalletError::Failed {
                        msg: format!("wrong number of memos (expected {})", txn.txn.output_len()),
                    });
                }
            }
            CapeTransition::Transaction(CapeModelTxn::Burn { xfr, .. }) => {
                if TransactionNote::Transfer(Box::new(*xfr.clone()))
                    .verify_receiver_memos_signature(&memos, &sig)
                    .is_err()
                {
                    return Err(CapeWalletError::Failed {
                        msg: String::from("invalid memos signature"),
                    });
                }
                if memos.len() != txn.txn.output_len() {
                    return Err(CapeWalletError::Failed {
                        msg: format!("wrong number of memos (expected {})", txn.txn.output_len()),
                    });
                }
            }
            _ => {
                return Err(CapeWalletError::Failed {
                    msg: String::from("cannot post memos for wrap transactions"),
                });
            }
        }

        // Authenticate the validity of the records corresponding to the memos.
        let merkle_tree = &self.records;
        let merkle_paths = txn
            .uids
            .iter()
            .map(|uid| merkle_tree.get_leaf(*uid).expect_ok().unwrap().1.path)
            .collect::<Vec<_>>();

        // Store and broadcast the new memos.
        let memos = izip!(
            memos,
            txn.txn.output_commitments(),
            txn.uids.iter().cloned(),
            merkle_paths
        )
        .collect::<Vec<_>>();
        txn.memos = Some((memos.clone(), sig));
        let event = LedgerEvent::Memos {
            outputs: memos,
            transaction: Some((
                block_id as u64,
                txn_id as u64,
                txn.txn.hash(),
                txn.txn.kind(),
            )),
        };
        self.generate_event(event);

        Ok(())
    }

    fn memos_source(&self) -> EventSource {
        EventSource::QueryService
    }

    fn generate_event(&mut self, event: LedgerEvent<CapeLedger>) {
        self.events.publish(event)
    }

    fn event(
        &self,
        index: EventIndex,
        source: EventSource,
    ) -> Result<LedgerEvent<CapeLedger>, WalletError<CapeLedger>> {
        match source {
            EventSource::QueryService => self.events.get(index),
            _ => Err(WalletError::Failed {
                msg: String::from("invalid event source"),
            }),
        }
    }
}

pub type MockCapeLedger<'a> =
    MockLedger<'a, CapeLedger, MockCapeNetwork, AtomicWalletStorage<'a, CapeLedger, ()>>;

pub struct MockCapeBackend<'a, Meta: Serialize + DeserializeOwned> {
    storage: Arc<Mutex<AtomicWalletStorage<'a, CapeLedger, Meta>>>,
    pub(crate) ledger: Arc<Mutex<MockCapeLedger<'a>>>,
    key_stream: KeyTree,
}

impl<'a, Meta: Serialize + DeserializeOwned + Send + Clone + PartialEq> MockCapeBackend<'a, Meta> {
    pub fn new(
        ledger: Arc<Mutex<MockCapeLedger<'a>>>,
        loader: &mut impl WalletLoader<CapeLedger, Meta = Meta>,
    ) -> Result<MockCapeBackend<'a, Meta>, WalletError<CapeLedger>> {
        let storage = AtomicWalletStorage::new(loader, 1024)?;
        Ok(Self {
            key_stream: storage.key_stream(),
            storage: Arc::new(Mutex::new(storage)),
            ledger,
        })
    }

    pub fn new_for_test(
        ledger: Arc<Mutex<MockCapeLedger<'a>>>,
        storage: Arc<Mutex<AtomicWalletStorage<'a, CapeLedger, Meta>>>,
        key_stream: KeyTree,
    ) -> Result<MockCapeBackend<'a, Meta>, WalletError<CapeLedger>> {
        Ok(Self {
            key_stream,
            storage,
            ledger,
        })
    }
}

#[async_trait]
impl<'a, Meta: Serialize + DeserializeOwned + Send> WalletBackend<'a, CapeLedger>
    for MockCapeBackend<'a, Meta>
{
    type EventStream = Pin<Box<dyn Stream<Item = (LedgerEvent<CapeLedger>, EventSource)> + Send>>;
    type Storage = AtomicWalletStorage<'a, CapeLedger, Meta>;

    async fn storage<'l>(&'l mut self) -> MutexGuard<'l, Self::Storage> {
        self.storage.lock().await
    }

    async fn create(&mut self) -> Result<WalletState<'a, CapeLedger>, WalletError<CapeLedger>> {
        let univ_param = &*UNIVERSAL_PARAM;
        let state = self
            .ledger
            .lock()
            .await
            .network()
            .create_wallet(univ_param)?;
        self.storage().await.create(&state).await?;
        Ok(state)
    }

    async fn subscribe(&self, from: EventIndex, to: Option<EventIndex>) -> Self::EventStream {
        self.ledger.lock().await.network().subscribe(from, to)
    }

    async fn get_public_key(
        &self,
        address: &UserAddress,
    ) -> Result<UserPubKey, WalletError<CapeLedger>> {
        self.ledger.lock().await.network().get_public_key(address)
    }

    async fn register_user_key(
        &mut self,
        key_pair: &UserKeyPair,
    ) -> Result<(), WalletError<CapeLedger>> {
        self.ledger
            .lock()
            .await
            .network()
            .register_user_key(key_pair)
    }

    async fn get_initial_scan_state(
        &self,
        _from: EventIndex,
    ) -> Result<(MerkleTree, EventIndex), CapeWalletError> {
        self.ledger.lock().await.get_initial_scan_state()
    }

    async fn get_nullifier_proof(
        &self,
        nullifiers: &mut CapeNullifierSet,
        nullifier: Nullifier,
    ) -> Result<(bool, ()), WalletError<CapeLedger>> {
        // Try to look up the nullifier in our "local" cache. If it is not there, query the contract
        // and cache it.
        match nullifiers.get(nullifier) {
            Some(ret) => Ok((ret, ())),
            None => {
                let ret = self
                    .ledger
                    .lock()
                    .await
                    .network()
                    .nullifier_spent(nullifier);
                nullifiers.insert(nullifier, ret);
                Ok((ret, ()))
            }
        }
    }

    fn key_stream(&self) -> KeyTree {
        self.key_stream.clone()
    }

    async fn submit(
        &mut self,
        txn: CapeTransition,
        info: TransactionInfo<CapeLedger>,
    ) -> Result<(), WalletError<CapeLedger>> {
        let mut ledger = self.ledger.lock().await;
        ledger.network().store_call_data(
            info.uid.unwrap_or_else(|| TransactionUID(txn.hash())),
            info.memos.into_iter().flatten().collect(),
            info.sig,
        );
        ledger.submit(txn)
    }
}

#[async_trait]
impl<'a, Meta: Serialize + DeserializeOwned + Send> CapeWalletBackend<'a>
    for MockCapeBackend<'a, Meta>
{
    async fn register_erc20_asset(
        &mut self,
        asset: &AssetDefinition,
        erc20_code: Erc20Code,
        sponsor: EthereumAddr,
    ) -> Result<(), WalletError<CapeLedger>> {
        self.ledger
            .lock()
            .await
            .network()
            .register_erc20(asset.clone(), erc20_code, sponsor)
            .map_err(cape_to_wallet_err)
    }

    async fn get_wrapped_erc20_code(
        &self,
        asset: &AssetCode,
    ) -> Result<Option<Erc20Code>, WalletError<CapeLedger>> {
        self.ledger.lock().await.network().get_wrapped_asset(asset)
    }

    async fn wait_for_wrapped_erc20_code(
        &mut self,
        asset: &AssetCode,
        timeout: Option<Duration>,
    ) -> Result<(), CapeWalletError> {
        let mut backoff = Duration::from_secs(1);
        let now = Instant::now();
        while self.get_wrapped_erc20_code(asset).await?.is_none() {
            if let Some(time) = timeout {
                if now.elapsed() >= time {
                    return Err(CapeWalletError::Failed {
                        msg: format!("asset not reflected in the EQS in {:?}", time),
                    });
                }
            }
            sleep(backoff).await;
            backoff = min(backoff * 2, Duration::from_secs(60));
        }
        Ok(())
    }

    async fn wrap_erc20(
        &mut self,
        erc20_code: Erc20Code,
        src_addr: EthereumAddr,
        ro: RecordOpening,
    ) -> Result<(), WalletError<CapeLedger>> {
        self.ledger
            .lock()
            .await
            .network()
            .wrap_erc20(erc20_code, src_addr, ro)
            .map_err(cape_to_wallet_err)
    }

    fn eth_client(&self) -> Result<Arc<EthMiddleware>, CapeWalletError> {
        Err(CapeWalletError::Failed {
            msg: String::from("eth_client is not implemented for MockCapeBackend"),
        })
    }

    fn asset_verifier(&self) -> VerKey {
        test_asset_signing_key().ver_key()
    }

    async fn eqs_time(&self) -> Result<EventIndex, CapeWalletError> {
        Ok(self.ledger.lock().await.network().events.now())
    }

    async fn wait_for_eqs(&self) -> Result<(), CapeWalletError> {
        // No need to wait for the mock EQS.
        Ok(())
    }

    async fn contract_address(&self) -> Result<Erc20Code, CapeWalletError> {
        // We're mocking a deployment of the CAPE contract, so the address can be anything we want.
        Ok(Erc20Code::default())
    }

    async fn latest_contract_address(&self) -> Result<Erc20Code, CapeWalletError> {
        // This just has to match `contract_address`, so that the contract appears up to date.
        Ok(Erc20Code::default())
    }
}

fn cape_to_wallet_err(err: CapeValidationError) -> WalletError<CapeLedger> {
    //TODO Convert CapeValidationError to WalletError in a better way. Maybe WalletError should be
    // parameterized on the ledger type and there should be a ledger trait ValidationError.
    WalletError::Failed {
        msg: err.to_string(),
    }
}

/// A mock CAPE wallet backend that can be used to replay a given list of events.
pub struct ReplayBackend<'a, Meta: Send + DeserializeOwned + Serialize> {
    events: Vec<LedgerEvent<CapeLedger>>,
    storage: Arc<Mutex<AtomicWalletStorage<'a, CapeLedger, Meta>>>,
    key_stream: KeyTree,
}

impl<'a, Meta: Clone + PartialEq + Send + DeserializeOwned + Serialize> ReplayBackend<'a, Meta> {
    pub fn new(
        events: Vec<LedgerEvent<CapeLedger>>,
        loader: &mut impl WalletLoader<CapeLedger, Meta = Meta>,
    ) -> Self {
        let storage = AtomicWalletStorage::new(loader, 1024).unwrap();
        let key_stream = storage.key_stream();
        Self {
            events,
            storage: Arc::new(Mutex::new(storage)),
            key_stream,
        }
    }
}

#[async_trait]
impl<'a, Meta: Send + DeserializeOwned + Serialize> WalletBackend<'a, CapeLedger>
    for ReplayBackend<'a, Meta>
{
    type EventStream = Pin<Box<dyn Stream<Item = (LedgerEvent<CapeLedger>, EventSource)> + Send>>;
    type Storage = AtomicWalletStorage<'a, CapeLedger, Meta>;

    async fn storage<'l>(&'l mut self) -> MutexGuard<'l, Self::Storage> {
        self.storage.lock().await
    }

    async fn create(&mut self) -> Result<WalletState<'a, CapeLedger>, WalletError<CapeLedger>> {
        // Get the state from the snapshotted time (i.e. after replaying all the events).
        let mut record_mt = MerkleTree::new(CAPE_MERKLE_HEIGHT).unwrap();
        let mut block_height = 0;
        for event in &self.events {
            if let LedgerEvent::Commit { block, .. } = event {
                for txn in block.txns() {
                    for comm in txn.output_commitments() {
                        record_mt.push(comm.to_field_element());
                    }
                }
                block_height += 1;
            }
        }

        // `record_mt` should be completely sparse.
        for uid in 0..record_mt.num_leaves() - 1 {
            record_mt.forget(uid);
        }
        let merkle_leaf_to_forget = if record_mt.num_leaves() > 0 {
            Some(record_mt.num_leaves() - 1)
        } else {
            None
        };

        Ok(WalletState {
            proving_keys: Arc::new(crate::backend::gen_proving_keys(&*UNIVERSAL_PARAM)),
            txn_state: TransactionState {
                validator: CapeTruster::new(block_height, record_mt.num_leaves()),
                now: EventIndex::from_source(EventSource::QueryService, self.events.len()),
                // Completely sparse nullifier set.
                nullifiers: Default::default(),
                record_mt,
                records: RecordDatabase::default(),
                merkle_leaf_to_forget,
                transactions: Default::default(),
            },
            key_state: Default::default(),
            assets: Default::default(),
            viewing_accounts: Default::default(),
            freezing_accounts: Default::default(),
            sending_accounts: Default::default(),
        })
    }

    async fn subscribe(&self, from: EventIndex, to: Option<EventIndex>) -> Self::EventStream {
        let from = from.index(EventSource::QueryService);
        let to = to.map(|to| to.index(EventSource::QueryService));

        println!(
            "playing back {} events from {} to {:?}",
            self.events.len(),
            from,
            to
        );
        let events = iter(self.events.clone())
            .enumerate()
            .map(|(i, event)| {
                println!("replaying event {} {:?}", i, event);
                (event, EventSource::QueryService)
            })
            .skip(from);
        let events: Self::EventStream = if let Some(to) = to {
            Box::pin(events.take(to - from))
        } else {
            Box::pin(events)
        };
        // Append a stream which blocks forever, since the event stream is not supposed to terminate.
        Box::pin(events.chain(pending()))
    }

    async fn get_public_key(
        &self,
        _address: &UserAddress,
    ) -> Result<UserPubKey, WalletError<CapeLedger>> {
        // Since we're not generating transactions, we don't need to support address book queries.
        Err(WalletError::Failed {
            msg: "address book not supported".into(),
        })
    }

    async fn register_user_key(
        &mut self,
        _key_pair: &UserKeyPair,
    ) -> Result<(), WalletError<CapeLedger>> {
        // The wallet calls this function when it generates a key, so it has to succeed, but since
        // we don't support querying the address book, it doesn't have to do anything.
        Ok(())
    }

    async fn get_initial_scan_state(
        &self,
        _from: EventIndex,
    ) -> Result<(MerkleTree, EventIndex), CapeWalletError> {
        Ok((
            MerkleTree::new(CAPE_MERKLE_HEIGHT).unwrap(),
            EventIndex::default(),
        ))
    }

    async fn get_nullifier_proof(
        &self,
        _nullifiers: &mut CapeNullifierSet,
        _nullifier: Nullifier,
    ) -> Result<(bool, ()), WalletError<CapeLedger>> {
        // Nullifier queries are not needed for event playback.
        Err(WalletError::Failed {
            msg: "nullifier queries not supported".into(),
        })
    }

    fn key_stream(&self) -> KeyTree {
        self.key_stream.clone()
    }

    async fn submit(
        &mut self,
        _txn: CapeTransition,
        _info: TransactionInfo<CapeLedger>,
    ) -> Result<(), WalletError<CapeLedger>> {
        Err(WalletError::Failed {
            msg: "transacting not supported".into(),
        })
    }
}

pub struct MockCapeWalletLoader {
    pub path: PathBuf,
    pub key: KeyTree,
}

impl WalletLoader<CapeLedger> for MockCapeWalletLoader {
    type Meta = ();

    fn location(&self) -> PathBuf {
        self.path.clone()
    }

    fn create(&mut self) -> Result<(Self::Meta, KeyTree), WalletError<CapeLedger>> {
        Ok(((), self.key.clone()))
    }

    fn load(&mut self, _meta: &mut Self::Meta) -> Result<KeyTree, WalletError<CapeLedger>> {
        Ok(self.key.clone())
    }
}

pub struct CapeTest {
    rng: ChaChaRng,
    temp_dirs: Vec<TempDir>,
}

impl CapeTest {
    fn temp_dir(&mut self) -> PathBuf {
        let dir = TempDir::new("cape_wallet").unwrap();
        let path = PathBuf::from(dir.path());
        self.temp_dirs.push(dir);
        path
    }
}

impl Default for CapeTest {
    fn default() -> Self {
        Self {
            rng: ChaChaRng::from_seed([42u8; 32]),
            temp_dirs: Vec::new(),
        }
    }
}

#[async_trait]
impl<'a> SystemUnderTest<'a> for CapeTest {
    type Ledger = CapeLedger;
    type MockBackend = MockCapeBackend<'a, ()>;
    type MockNetwork = MockCapeNetwork;
    type MockStorage = AtomicWalletStorage<'a, CapeLedger, ()>;

    async fn create_network(
        &mut self,
        verif_crs: VerifierKeySet,
        _proof_crs: ProverKeySet<'a, OrderByOutputs>,
        records: MerkleTree,
        initial_grants: Vec<(RecordOpening, u64)>,
    ) -> Self::MockNetwork {
        let initial_memos = initial_grants
            .into_iter()
            .map(|(ro, uid)| (ReceiverMemo::from_ro(&mut self.rng, &ro, &[]).unwrap(), uid))
            .collect();
        MockCapeNetwork::new(verif_crs, records, initial_memos)
    }

    async fn create_storage(&mut self) -> Self::MockStorage {
        let mut loader = MockCapeWalletLoader {
            path: self.temp_dir(),
            key: KeyTree::random(&mut self.rng).0,
        };
        AtomicWalletStorage::new(&mut loader, 128).unwrap()
    }

    async fn create_backend(
        &mut self,
        ledger: Arc<Mutex<MockLedger<'a, Self::Ledger, Self::MockNetwork, Self::MockStorage>>>,
        _initial_grants: Vec<(RecordOpening, u64)>,
        key_stream: KeyTree,
        storage: Arc<Mutex<Self::MockStorage>>,
    ) -> Self::MockBackend {
        MockCapeBackend::new_for_test(ledger, storage, key_stream).unwrap()
    }
}

// CAPE-specific tests
#[cfg(test)]
mod cape_wallet_tests {
    use super::*;
    use crate::wallet::CapeWalletExt;
    use jf_cap::structs::{AssetCode, AssetPolicy};
    use seahorse::{txn_builder::TransactionError, RecordAmount};
    use std::time::Instant;

    #[cfg(feature = "slow-tests")]
    use testing::generic_wallet_tests;
    #[cfg(feature = "slow-tests")]
    seahorse::instantiate_generic_wallet_tests!(CapeTest);

    #[async_std::test]
    async fn test_cape_wallet() -> std::io::Result<()> {
        let mut t = CapeTest::default();

        // Initialize a ledger and wallet, and get the owner address.
        let mut now = Instant::now();
        let num_inputs = 2;
        let num_outputs = 2;
        let total_initial_grant = 20u64;
        let initial_grant = RecordAmount::from(total_initial_grant / 2);
        let (ledger, mut wallets) = t
            .create_test_network(
                &[(num_inputs, num_outputs)],
                vec![total_initial_grant],
                &mut now,
            )
            .await;
        assert_eq!(wallets.len(), 1);
        let owner = wallets[0].1[0].clone();
        t.sync(&ledger, &wallets).await;
        println!("CAPE wallet created: {}s", now.elapsed().as_secs_f32());

        // Check the balance after CAPE wallet initialization.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &AssetCode::native())
                .await,
            u128::from(initial_grant).into()
        );

        // Create an ERC20 code, sponsor address, and asset information.
        now = Instant::now();
        let erc20_addr = EthereumAddr([1u8; 20]);
        let erc20_code = Erc20Code(erc20_addr);
        let sponsor_addr = EthereumAddr([2u8; 20]);
        let cap_asset_policy = AssetPolicy::default();

        // Sponsor the ERC20 token.
        let cap_asset = wallets[0]
            .0
            .sponsor(
                "sponsored_asset".into(),
                erc20_code,
                sponsor_addr.clone(),
                cap_asset_policy,
            )
            .await
            .unwrap();
        println!("Sponsor completed: {}s", now.elapsed().as_secs_f32());

        // Check that the sponsored asset is added to the asset library.
        let info = wallets[0].0.asset(cap_asset.code).await.unwrap();
        assert_eq!(info.definition, cap_asset);
        assert_eq!(info.name, Some("sponsored_asset".into()));

        // Wrapping an undefined asset should fail.
        let wrap_amount = RecordAmount::from(6u64);
        match wallets[0]
            .0
            .wrap(
                sponsor_addr.clone(),
                AssetDefinition::dummy(),
                owner.clone(),
                wrap_amount,
            )
            .await
        {
            Err(WalletError::UndefinedAsset { asset: _ }) => {}
            e => {
                panic!("Expected WalletError::UndefinedAsset, found {:?}", e);
            }
        };

        // Wrap the sponsored asset.
        now = Instant::now();
        wallets[0]
            .0
            .wrap(
                sponsor_addr.clone(),
                cap_asset.clone(),
                owner.clone(),
                wrap_amount,
            )
            .await
            .unwrap();
        println!("Wrap completed: {}s", now.elapsed().as_secs_f32());
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            0u64.into()
        );

        // Submit dummy transactions to finalize the wrap.
        now = Instant::now();
        let dummy_coin = wallets[0]
            .0
            .define_asset(
                "defined_asset".into(),
                "Dummy asset".as_bytes(),
                Default::default(),
            )
            .await
            .unwrap();
        let mint_fee = RecordAmount::from(1u64);
        wallets[0]
            .0
            .mint(
                Some(&owner),
                mint_fee,
                &dummy_coin.code,
                5u64,
                owner.clone(),
            )
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!(
            "Dummy transactions submitted and wrap finalized: {}s",
            now.elapsed().as_secs_f32()
        );

        // Check the balance after the wrap.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &AssetCode::native())
                .await,
            (initial_grant - mint_fee).into()
        );
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            wrap_amount.into()
        );

        // Burning an amount more than the wrapped asset should fail.
        let mut burn_amount = wrap_amount + 1u64.into();
        let burn_fee = RecordAmount::from(1u64);
        match wallets[0]
            .0
            .burn(
                Some(&owner),
                sponsor_addr.clone(),
                &cap_asset.code.clone(),
                burn_amount,
                burn_fee,
            )
            .await
        {
            Err(WalletError::TransactionError {
                source: TransactionError::InsufficientBalance { .. },
            }) => {}
            e => {
                panic!(
                    "Expected TransactionError::InsufficientBalance, found {:?}",
                    e
                );
            }
        }

        // Burn the wrapped asset.
        now = Instant::now();
        burn_amount = wrap_amount;
        wallets[0]
            .0
            .burn(
                Some(&owner),
                sponsor_addr.clone(),
                &cap_asset.code.clone(),
                burn_amount,
                burn_fee,
            )
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!("Burn completed: {}s", now.elapsed().as_secs_f32());

        // Check the balance after the burn.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            0u64.into()
        );
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &AssetCode::native())
                .await,
            // TODO should we implement From<Amount> for U256 in jf-cap?
            u128::from(initial_grant - mint_fee - burn_fee).into(),
        );

        Ok(())
    }

    // Test a burn where the fee and the wrapped asset come from different accounts.
    #[cfg(feature = "slow-tests")]
    #[async_std::test]
    async fn test_burn_separate_accounts() -> std::io::Result<()> {
        let mut t = CapeTest::default();

        // Initialize a ledger and wallet, and get the owner address.
        let mut now = Instant::now();
        let (ledger, mut wallets) = t
            .create_test_network(
                &[(1, 2), (2, 2), (2, 3), (3, 3)],
                vec![20u64.into()],
                &mut now,
            )
            .await;

        // Create an ERC20 code, sponsor address, and asset information.
        let erc20_addr = EthereumAddr([1u8; 20]);
        let erc20_code = Erc20Code(erc20_addr);
        let sponsor_addr = EthereumAddr([2u8; 20]);
        let cap_asset_policy = AssetPolicy::default();

        // Sponsor the ERC20 token.
        now = Instant::now();
        let cap_asset = wallets[0]
            .0
            .sponsor(
                "sponsored_asset".into(),
                erc20_code,
                sponsor_addr.clone(),
                cap_asset_policy,
            )
            .await
            .unwrap();
        println!("Sponsor completed: {}s", now.elapsed().as_secs_f32());

        // Wrap the sponsored asset, to a fresh account with no native tokens. This will force the
        // wallet to use different accounts for the burn record and burn fee.
        now = Instant::now();
        let wrap_account = wallets[0]
            .0
            .generate_user_key("wrap account".into(), None)
            .await
            .unwrap()
            .address();
        wallets[0]
            .0
            .wrap(
                sponsor_addr.clone(),
                cap_asset.clone(),
                wrap_account.clone(),
                5,
            )
            .await
            .unwrap();
        println!("Wrap completed: {}s", now.elapsed().as_secs_f32());

        // Submit dummy transactions to finalize the wrap.
        now = Instant::now();
        let dummy_coin = wallets[0]
            .0
            .define_asset(
                "defined_asset".into(),
                "Dummy asset".as_bytes(),
                Default::default(),
            )
            .await
            .unwrap();
        let fee_account = wallets[0].1[0].clone();
        wallets[0]
            .0
            .mint(
                Some(&fee_account),
                1,
                &dummy_coin.code,
                5,
                fee_account.clone(),
            )
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!(
            "Dummy transactions submitted and wrap finalized: {}s",
            now.elapsed().as_secs_f32()
        );

        // Check the balance after the wrap.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&wrap_account, &cap_asset.code)
                .await,
            5u64.into()
        );
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&wrap_account, &AssetCode::native())
                .await,
            0u64.into()
        );

        // Burn the wrapped asset.
        now = Instant::now();
        wallets[0]
            .0
            .burn(None, sponsor_addr.clone(), &cap_asset.code.clone(), 5, 1)
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!("Burn completed: {}s", now.elapsed().as_secs_f32());

        // Check the balance of the wrapper account after the burn. We don't know the balance of the
        // fee account, because the fee could have been paid from either of the preinitialized
        // native token accounts.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&wrap_account, &cap_asset.code)
                .await,
            0u64.into()
        );

        Ok(())
    }

    // Test a burn with a change output.
    #[cfg(feature = "slow-tests")]
    #[async_std::test]
    async fn test_burn_change() -> std::io::Result<()> {
        let mut t = CapeTest::default();

        // Initialize a ledger and wallet, and get the owner address.
        let mut now = Instant::now();
        let (ledger, mut wallets) = t
            .create_test_network(
                &[(1, 2), (2, 2), (2, 3), (3, 3)],
                vec![20u64.into()],
                &mut now,
            )
            .await;

        // Create an ERC20 code, sponsor address, and asset information.
        let erc20_addr = EthereumAddr([1u8; 20]);
        let erc20_code = Erc20Code(erc20_addr);
        let sponsor_addr = EthereumAddr([2u8; 20]);
        let cap_asset_policy = AssetPolicy::default();

        // Sponsor the ERC20 token.
        now = Instant::now();
        let cap_asset = wallets[0]
            .0
            .sponsor(
                "sponsored_asset".into(),
                erc20_code,
                sponsor_addr.clone(),
                cap_asset_policy,
            )
            .await
            .unwrap();
        println!("Sponsor completed: {}s", now.elapsed().as_secs_f32());

        // Wrap the sponsored asset.
        now = Instant::now();
        let owner = wallets[0].1[0].clone();
        wallets[0]
            .0
            .wrap(sponsor_addr.clone(), cap_asset.clone(), owner.clone(), 5)
            .await
            .unwrap();
        println!("Wrap completed: {}s", now.elapsed().as_secs_f32());

        // Submit dummy transactions to finalize the wrap.
        now = Instant::now();
        let dummy_coin = wallets[0]
            .0
            .define_asset(
                "defined_asset".into(),
                "Dummy asset".as_bytes(),
                Default::default(),
            )
            .await
            .unwrap();
        wallets[0]
            .0
            .mint(Some(&owner), 1, &dummy_coin.code, 5, owner.clone())
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!(
            "Dummy transactions submitted and wrap finalized: {}s",
            now.elapsed().as_secs_f32()
        );

        // Check the balance after the wrap.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            5u64.into()
        );

        // Burn the wrapped asset.
        now = Instant::now();
        wallets[0]
            .0
            .burn(None, sponsor_addr.clone(), &cap_asset.code.clone(), 3, 1)
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!("Burn completed: {}s", now.elapsed().as_secs_f32());

        // Check the balance after the burn.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            2u64.into()
        );

        Ok(())
    }

    // Test a burn where the input is aggegated from multiple records.
    #[cfg(feature = "slow-tests")]
    #[async_std::test]
    async fn test_burn_aggregate() -> std::io::Result<()> {
        let mut t = CapeTest::default();

        // Initialize a ledger and wallet, and get the owner address.
        let mut now = Instant::now();
        let (ledger, mut wallets) = t
            .create_test_network(
                &[(1, 2), (2, 2), (2, 3), (3, 3)],
                vec![20u64.into()],
                &mut now,
            )
            .await;

        // Create an ERC20 code, sponsor address, and asset information.
        let erc20_addr = EthereumAddr([1u8; 20]);
        let erc20_code = Erc20Code(erc20_addr);
        let sponsor_addr = EthereumAddr([2u8; 20]);
        let cap_asset_policy = AssetPolicy::default();

        // Sponsor the ERC20 token.
        now = Instant::now();
        let cap_asset = wallets[0]
            .0
            .sponsor(
                "sponsored_asset".into(),
                erc20_code,
                sponsor_addr.clone(),
                cap_asset_policy,
            )
            .await
            .unwrap();
        println!("Sponsor completed: {}s", now.elapsed().as_secs_f32());

        // Wrap two records of the sponsored asset.
        now = Instant::now();
        let owner = wallets[0].1[0].clone();
        wallets[0]
            .0
            .wrap(sponsor_addr.clone(), cap_asset.clone(), owner.clone(), 2)
            .await
            .unwrap();
        wallets[0]
            .0
            .wrap(sponsor_addr.clone(), cap_asset.clone(), owner.clone(), 3)
            .await
            .unwrap();
        println!("Wraps completed: {}s", now.elapsed().as_secs_f32());

        // Submit dummy transactions to finalize the wrap.
        now = Instant::now();
        let dummy_coin = wallets[0]
            .0
            .define_asset(
                "defined_asset".into(),
                "Dummy asset".as_bytes(),
                Default::default(),
            )
            .await
            .unwrap();
        wallets[0]
            .0
            .mint(Some(&owner), 1, &dummy_coin.code, 5, owner.clone())
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!(
            "Dummy transactions submitted and wrap finalized: {}s",
            now.elapsed().as_secs_f32()
        );

        // Check the balance after the wrap.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            5u64.into()
        );

        // Burn the wrapped asset.
        now = Instant::now();
        wallets[0]
            .0
            .burn(None, sponsor_addr.clone(), &cap_asset.code.clone(), 5, 1)
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!("Burn completed: {}s", now.elapsed().as_secs_f32());

        // Check the balance after the burn.
        assert_eq!(
            wallets[0]
                .0
                .balance_breakdown(&owner, &cap_asset.code)
                .await,
            0u64.into()
        );

        Ok(())
    }

    #[cfg(feature = "slow-tests")]
    #[async_std::test]
    async fn test_unwrap_viewing() {
        let mut t = CapeTest::default();

        // Initialize a ledger and wallet, and get the owner address.
        let mut now = Instant::now();
        let (ledger, mut wallets) = t
            .create_test_network(
                &[(1, 2), (2, 2), (2, 3), (3, 3)],
                vec![20u64.into()],
                &mut now,
            )
            .await;

        // Create an ERC20 code, sponsor address, and asset information.
        let erc20_addr = EthereumAddr([1u8; 20]);
        let erc20_code = Erc20Code(erc20_addr);
        let sponsor_addr = EthereumAddr([2u8; 20]);
        let viewing_key = wallets[0]
            .0
            .generate_audit_key("viewing".into())
            .await
            .unwrap();
        let freezing_key = wallets[0]
            .0
            .generate_freeze_key("freezing".into())
            .await
            .unwrap();
        let cap_asset_policy = AssetPolicy::default()
            .set_auditor_pub_key(viewing_key)
            .reveal_record_opening()
            .unwrap()
            .set_freezer_pub_key(freezing_key);

        // Sponsor the ERC20 token.
        now = Instant::now();
        let cap_asset = wallets[0]
            .0
            .sponsor(
                "sponsored_asset".into(),
                erc20_code,
                sponsor_addr.clone(),
                cap_asset_policy,
            )
            .await
            .unwrap();
        println!("Sponsor completed: {}s", now.elapsed().as_secs_f32());

        // Wrap the sponsored asset.
        let owner = wallets[0].1[0].clone();
        wallets[0]
            .0
            .wrap(sponsor_addr.clone(), cap_asset.clone(), owner.clone(), 3)
            .await
            .unwrap();

        // Submit dummy transactions to finalize the wrap.
        now = Instant::now();
        wallets[0]
            .0
            .transfer(None, &AssetCode::native(), &[(owner.clone(), 1)], 1)
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        println!(
            "Dummy transactions submitted and wrap finalized: {}s",
            now.elapsed().as_secs_f32()
        );
        assert_eq!(wallets[0].0.balance(&cap_asset.code).await, 3u64.into());

        // Unwrap with change.
        wallets[0]
            .0
            .burn(None, sponsor_addr.clone(), &cap_asset.code, 2, 0)
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        assert_eq!(wallets[0].0.balance(&cap_asset.code).await, 1u64.into());

        // We should be able to view the change record from the unwrap, but _not_ the burned record.
        let records = wallets[0]
            .0
            .records()
            .await
            .filter(|rec| rec.ro.asset_def.code == cap_asset.code)
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].ro.amount, 1u64.into());

        // Unwrap again just to make sure things still work.
        wallets[0]
            .0
            .burn(None, sponsor_addr, &cap_asset.code, 1, 0)
            .await
            .unwrap();
        t.sync(&ledger, &wallets).await;
        assert_eq!(wallets[0].0.balance(&cap_asset.code).await, 0u64.into());
    }
}
