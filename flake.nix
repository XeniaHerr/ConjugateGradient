{
  description = "Beginner Practical";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
      pkgs = import nixpkgs { system = "${system}"; config.allowUnfree = true; }; in
      {
        devShells.default = pkgs.mkShell {

    buildInputs = with pkgs; [
      adaptivecppWithCuda
      cmake
      bear
      doxygen
      boost
      openmpi
      #tinyxml2
      tinyxml-2
        
            paraview
            valgrind


      pkg-config

    ];
        };


      }
    );
}
