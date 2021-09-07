with import ./nix/nixpkgs.nix { };

let
  geth = callPackage ./nix/go-ethereum {
    inherit (darwin) libobjc;
    inherit (darwin.apple_sdk.frameworks) IOKit;
  };
in
mkShell
{
  buildInputs = [
    geth
    nodePackages.pnpm
    solc # solidity compiler
    hivemind # process runner
    nodejs-12_x # nodejs
  ];
  shellHook = ''
    export PATH=$(pwd)/bin:$(pwd)/node_modules/.bin:$PATH
  '';
}
