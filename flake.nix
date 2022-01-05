{
  description = "A devShell example";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  inputs.flake-compat.url = "github:edolstra/flake-compat";
  inputs.flake-compat.flake = false;

  inputs.fenix.url = "github:nix-community/fenix";
  inputs.fenix.inputs.nixpkgs.follows = "nixpkgs";

  inputs.crate2nix.url = "github:balsoft/crate2nix/balsoft/fix-broken-ifd";
  inputs.crate2nix.flake = false;

  inputs.pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  # See https://github.com/cachix/pre-commit-hooks.nix/pull/122
  inputs.pre-commit-hooks.inputs.flake-utils.follows = "flake-utils";
  inputs.pre-commit-hooks.inputs.nixpkgs.follows = "nixpkgs";

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , flake-compat
    , fenix
    , crate2nix
    , pre-commit-hooks
    , ...
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      fenixPackage = fenix.packages.${system}.stable.withComponents [ "cargo" "clippy" "rust-src" "rustc" "rustfmt" ];
      rustOverlay = final: prev:
        {
          inherit fenixPackage;
          rustc = fenixPackage;
          cargo = fenixPackage;
          rust-src = fenixPackage;
        };

      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          rustOverlay
        ];
      };
      checks = {
        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            lint-solidity = {
              enable = true;
              files = "^contracts/contracts/";
              entry = "lint-solidity";
              types = [ "solidity" ];
            };
            check-format = {
              enable = true;
              entry = "treefmt --fail-on-change";
            };
            # The hook "clippy" that ships with nix-precommit-hooks is outdated.
            cargo-clippy = {
              enable = true;
              description = "Lint Rust code.";
              entry = "cargo-clippy --workspace -- -D warnings";
              files = "\\.rs$";
              pass_filenames = false;
            };
          };
        };
      };

      inherit (import "${crate2nix}/tools.nix" { inherit pkgs; })
        generatedCargoNix;

      project = import
        (generatedCargoNix {
          name = "cape";
          src = ./.;
        })
        {
          inherit pkgs;
          defaultCrateOverrides = pkgs.defaultCrateOverrides // {
            # Crate dependency overrides go here
          };
        };

    in
    rec {


      inherit checks;

      # packages.cape = project.workspaceMembers.cape.build;
      # packages.tests.cape = project.workspaceMembers.cape.override {
      #   runTests = true;
      # };

      # defaultPackage = self.packages.${system}.workspaceMembers.cape;

      devShell =
        let
          mySolc = pkgs.callPackage ./nix/solc-bin { version = "0.8.10"; };
          pythonEnv = pkgs.poetry2nix.mkPoetryEnv {
            projectDir = ./.;
            overrides = pkgs.poetry2nix.overrides.withDefaults
              (import ./nix/poetryOverrides.nix { inherit pkgs; });
          };
          myPython = with pkgs; [
            poetry
            pythonEnv
          ];

          rustDeps = with pkgs; [
            pkgconfig
            openssl

            curl
            plantuml
            # cargo-edit

            fenixPackage
          ] ++ lib.optionals stdenv.isDarwin [
            # required to compile ethers-rs
            darwin.apple_sdk.frameworks.Security
            darwin.apple_sdk.frameworks.CoreFoundation

            # https://github.com/NixOS/nixpkgs/issues/126182
            libiconv
          ] ++ lib.optionals (stdenv.system != "aarch64-darwin") [
            # cargo-watch # broken: https://github.com/NixOS/nixpkgs/issues/146349
          ];
          # nixWithFlakes allows pre v2.4 nix installations to use flake commands (like `nix flake update`)
          nixWithFlakes = pkgs.writeShellScriptBin "nix" ''
            exec ${pkgs.nixFlakes}/bin/nix --experimental-features "nix-command flakes" "$@"
          '';
        in
        pkgs.mkShell
          {
            # inputsFrom = builtins.attrValues self.packages.${system};
            buildInputs = with pkgs; [
              nixWithFlakes
              go-ethereum
              nodePackages.pnpm
              mySolc
              hivemind # process runner
              nodejs-12_x # nodejs
              jq
              entr # watch files for changes, for example: ls contracts/*.sol | entr -c hardhat compile
              treefmt # multi language formatter
              nixpkgs-fmt
              git # required for pre-commit hook installation
              netcat-gnu # only used to check for open ports
              cacert
              mdbook # make-doc, documentation generation
              moreutils # includes `ts`, used to add timestamps on CI
            ]
            ++ myPython
            ++ rustDeps;

            RUST_SRC_PATH = "${fenixPackage}/lib/rustlib/src/rust/library";
            RUST_BACKTRACE = 1;
            RUST_LOG = "info";

            SOLCX_BINARY_PATH = "${mySolc}/bin";
            SOLC_VERSION = mySolc.version;
            SOLC_PATH = "${mySolc}/bin/solc";
            SOLC_OPTIMIZER_RUNS = "10000"; # TODO increase this once we have split up contract deployment

            shellHook = ''
              echo "Ensuring node dependencies are installed"
              pnpm --recursive install

              if [ ! -f .env ]; then
                echo "Copying .env.sample to .env"
                cp .env.sample .env
              fi

              echo "Exporting all vars in .env file"
              set -a; source .env; set +a;

              export CONTRACTS_DIR=$(pwd)/contracts
              export HARDHAT_CONFIG=$CONTRACTS_DIR/hardhat.config.ts
              export PATH=$(pwd)/node_modules/.bin:$PATH
              export PATH=$CONTRACTS_DIR/node_modules/.bin:$PATH
              export PATH=$(pwd)/bin:$PATH

              git config --local blame.ignoreRevsFile .git-blame-ignore-revs
            ''
            # install pre-commit hooks
            + self.checks.${system}.pre-commit-check.shellHook;
          };

    }
    );
}
