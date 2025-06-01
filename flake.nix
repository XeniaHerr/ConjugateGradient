{
  description = "Development Environment for my Beginner Practical";


  inputs = {


    nixpkgs.url = "github:NixOS/nixpkgs";

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs @ {self,...}: 

    inputs.flake-utils.lib.eachDefaultSystem( system:
  let
  pkgs = import inputs.nixpkgs { system = "${system}"; config.allowUnfree = true; };
      in
        {
        devShells.default = pkgs.mkShell {
          /*
            (python312.withPackages ( ps: with python312Packages; with ps; [
              matplotlib
              python312Packages.pandas
            ]))
*/
  buildInputs = with pkgs; [
            adaptivecppWithCuda
              patchelf
              file
              cmake
              python312Packages.matplotlib
              python312Packages.pandas
            python312
  ];
        };
      }
    );

  }
