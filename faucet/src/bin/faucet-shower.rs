use cap_rust_sandbox::universal_param::UNIVERSAL_PARAM;
use cape_wallet::{
    backend::{CapeBackend, CapeBackendConfig},
    wallet::{CapeWallet, CapeWalletError},
};
use ethers::prelude::Address;
use futures::stream::{iter, StreamExt};
use jf_cap::structs::AssetCode;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaChaRng,
};
use seahorse::{
    hd::KeyTree,
    loader::{Loader, LoaderMetadata},
    txn_builder::TransactionStatus,
};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Duration;
use structopt::StructOpt;
use surf::Url;
use tempdir::TempDir;

/// Turns a trickle into a shower.
///
/// Give faucet-shower a master mnemonic for a funded wallet and a number N and it will generate N
/// new wallets, transfer some tokens from the master wallet to each new wallet, and print the
/// mnemonics and public keys of the newly funded wallets.
#[derive(Debug, StructOpt)]
pub struct Options {
    /// mnemonic for the master faucet wallet
    #[structopt(short, long, env = "CAPE_FAUCET_WALLET_MNEMONIC")]
    pub master_mnemonic: String,

    /// number of new wallets to generate
    #[structopt(short, long, default_value = "10")]
    pub num_wallets: usize,

    /// size of grant for each new wallet
    #[structopt(short, long, default_value = "1000000")]
    pub transfer_size: u64,

    /// URL for the Ethereum Query Service.
    #[structopt(
        short,
        long,
        env = "CAPE_EQS_URL",
        default_value = "http://localhost:50087"
    )]
    pub eqs_url: Url,

    /// URL for the CAPE relayer.
    #[structopt(
        short,
        long,
        env = "CAPE_RELAYER_URL",
        default_value = "http://localhost:50077"
    )]
    pub relayer_url: Url,

    /// URL for the Ethereum Query Service.
    #[structopt(
        short,
        long,
        env = "CAPE_ADDRESS_BOOK_URL",
        default_value = "http://localhost:50078"
    )]
    pub address_book_url: Url,

    /// Address of the CAPE smart contract.
    #[structopt(short, long, env = "CAPE_CONTRACT_ADDRESS")]
    pub contract_address: Address,

    /// URL for Ethers HTTP Provider
    #[structopt(
        short,
        long,
        env = "CAPE_WEB3_PROVIDER_URL",
        default_value = "http://localhost:8545"
    )]
    pub rpc_url: Url,

    /// Minimum amount of time to wait between polling requests to EQS.
    #[structopt(long, env = "CAPE_WALLET_MIN_POLLING_DELAY", default_value = "500")]
    pub min_polling_delay_ms: u64,
}

async fn create_wallet(
    opt: &Options,
    rng: &mut ChaChaRng,
    mnemonic: String,
    dir: PathBuf,
) -> Result<CapeWallet<'static, CapeBackend<'static, LoaderMetadata>>, CapeWalletError> {
    // We are never going to re-open this wallet once it's created, so we don't really need a
    // password. Just make it random bytes.
    let mut password = [0; 32];
    rng.fill_bytes(&mut password);
    let mut loader = Loader::from_literal(Some(mnemonic), hex::encode(password), dir);
    let backend = CapeBackend::new(
        &*UNIVERSAL_PARAM,
        CapeBackendConfig {
            rpc_url: opt.rpc_url.clone(),
            eqs_url: opt.eqs_url.clone(),
            relayer_url: opt.relayer_url.clone(),
            address_book_url: opt.address_book_url.clone(),
            contract_address: opt.contract_address,
            // We're not going to do any direct-to-contract operations that would require an ETH
            // wallet. Everything we do will go through the relayer.
            eth_mnemonic: None,
            min_polling_delay: Duration::from_millis(opt.min_polling_delay_ms),
        },
        &mut loader,
    )
    .await?;
    CapeWallet::new(backend).await
}

#[async_std::main]
async fn main() {
    let opt = Options::from_args();
    let mut rng = ChaChaRng::from_entropy();
    let dir = TempDir::new("faucet-shower").unwrap();

    // Create the parent wallet.
    let parent_dir = [dir.path(), Path::new("parent")].iter().collect();
    let mut parent = create_wallet(&opt, &mut rng, opt.master_mnemonic.clone(), parent_dir)
        .await
        .unwrap();

    // Generate the key which will be used to transfer to the children. Tell it to start a scan
    // from the default index (the first event) so it can find records created by the faucet event.
    let parent_key = parent
        .generate_user_key("parent key".into(), Some(Default::default()))
        .await
        .unwrap();

    // While the ledger scan is going, create the child wallets.
    let children = iter(0..opt.num_wallets)
        .then(|i| {
            let mut rng = ChaChaRng::from_rng(&mut rng).unwrap();
            let dir = &dir;
            let opt = &opt;
            async move {
                let (_, mnemonic) = KeyTree::random(&mut rng);
                let dir = [dir.path(), Path::new(&format!("child_wallet_{}", i))]
                    .iter()
                    .collect();
                let mut wallet = create_wallet(opt, &mut rng, mnemonic.to_string(), dir)
                    .await
                    .unwrap();
                let key = wallet
                    .generate_user_key(format!("child key {}", i), None)
                    .await
                    .unwrap();
                (wallet, mnemonic, key)
            }
        })
        .collect::<Vec<_>>()
        .await;

    // Once we have all the wallets, we need to wait for the ledger scan so that the parent wallet
    // can discover a record to transfer from.
    parent.await_key_scan(&parent_key.address()).await.unwrap();
    let balance = parent.balance(&AssetCode::native()).await;
    if balance < opt.transfer_size * (opt.num_wallets as u64) {
        eprintln!(
            "Insufficient balance for transferring {} units to {} wallets: {}",
            opt.transfer_size, opt.num_wallets, balance
        );
        exit(1);
    }

    // Print out the generated child mnemonics and keys _before_ we start doing any transfers. If we
    // panic or get killed for any reason after we have transferred, it is crucial that we have
    // already reported all of the mnemonics needed to recover the funds.
    println!(
        "Transferring {} units each to the following wallets:",
        opt.transfer_size
    );
    for (_, mnemonic, key) in &children {
        println!("{} {}", mnemonic, key);
    }

    // Do the transfers.
    for (_, _, key) in &children {
        match parent
            .transfer(
                None,
                &AssetCode::native(),
                &[(key.address(), opt.transfer_size)],
                0,
            )
            .await
        {
            Ok(receipt) => match parent.await_transaction(&receipt).await {
                Ok(TransactionStatus::Retired) => {
                    println!("Transferred {} units to {}", opt.transfer_size, key)
                }
                Ok(status) => eprintln!(
                    "Transfer to {} did not complete successfully: {}",
                    key, status
                ),
                Err(err) => eprintln!("Error while waiting for transfer to {}: {}", key, err),
            },
            Err(err) => eprintln!("Failed to transfer to {}: {}", key, err),
        }
    }

    // Wait for the children to report the new balances.
    for (wallet, _, key) in &children {
        while wallet.balance(&AssetCode::native()).await < opt.transfer_size {
            eprintln!(
                "Waiting for {} to receive {} tokens",
                key, opt.transfer_size
            );
            async_std::task::sleep(Duration::from_secs(1)).await;
        }
    }
}