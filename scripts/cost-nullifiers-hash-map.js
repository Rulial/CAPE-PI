const { ethers } = require("hardhat");
const common = require("../lib/common");

async function print_report(owner, title, fun_to_eval, args) {
  console.log("**** " + title + "****");

  let res;
  try {
    res = await common.compute_gas_and_price(owner, fun_to_eval, args);
    let gas = res[0];

    let price = res[1].getValue();
    console.log(gas + " gas  ------ " + price + " USD ");
  } catch (error) {
    console.log(error);
  }

  console.log("\n");
}

async function main() {
  const [owner] = await ethers.getSigners();

  const AAPE = await ethers.getContractFactory("TestAAPE");
  const aape = await AAPE.deploy();

  // Polling interval in ms.
  aape.provider.pollingInterval = 20;

  await aape.deployed();

  console.log("Contract deployed at address " + aape.address);

  const NUM_MAX_NULLIFIERS = 10_000;

  for (let i = 1; i < NUM_MAX_NULLIFIERS; i += 1_000) {
    // Insert i nullifiers into the hash table
    for (let j = 0; j < i; j++) {
      // Insert nullifiers
      let nullifier = ethers.utils.randomBytes(32);
      await aape._insert_nullifier(nullifier);
    }

    // Measure how much it costs to check for membership
    let title = "Check for nullifiers. HASHMAP SIZE = " + i + " ";
    let test_nullifier = ethers.utils.randomBytes(32);
    await print_report(
      owner,
      title,
      aape._has_nullifier_already_been_published,
      [test_nullifier]
    );

    title = "Insert a nullifier. HASHMAP SIZE = " + i + " ";
    test_nullifier = ethers.utils.randomBytes(32);
    await print_report(owner, title, aape._insert_nullifier, [test_nullifier]);
  }
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });