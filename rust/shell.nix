{ pkgs ? import ../nix/nixpkgs.nix { } }:

with pkgs;

let
  stableToolchain = rust-bin.stable."1.56.0".minimal.override {
    extensions = [ "rustfmt" "clippy" "llvm-tools-preview" ];
  };
in
mkShell {

  buildInputs = [

    pkgconfig
    openssl

    curl

    stableToolchain

    cargo-edit
    cargo-watch
  ] ++ lib.optionals stdenv.isDarwin [
    # required to compile ethers-rs
    darwin.apple_sdk.frameworks.Security
    darwin.apple_sdk.frameworks.CoreFoundation

    # https://github.com/NixOS/nixpkgs/issues/126182
    libiconv
  ] ++ lib.optionals stdenv.isLinux [
    lld # a faster linker, does not work out of the box on OSX
  ];

  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  RUST_BACKTRACE = 1;
  RUSTFLAGS = if stdenv.isLinux then "-C link-arg=-fuse-ld=lld" else "";

  shellHook = ''
    export RUST_LOG=info

    # Needed with the ldd linker
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${openssl.out}/lib
  '';
}