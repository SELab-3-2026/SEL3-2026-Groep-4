{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-jetbrains-plugins.url = "github:theCapypara/nix-jetbrains-plugins";
  };

  outputs = { self, nixpkgs, flake-utils, nix-jetbrains-plugins }:
  flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    pluginList = [
      "be.ugent.piedcler.dodona"
      "com.github.copilot"
      "com.google.tools.ij.aiplugin"
      "IdeaVIM"
    ];
  in {
    devShells.default = pkgs.mkShell {
      packages = with pkgs; [
        python312
        python312Packages.uv

        ruff

        # Editor of your choice
        (nix-jetbrains-plugins.lib.buildIdeWithPlugins pkgs "pycharm" pluginList)
      ];
    };
  });
}
