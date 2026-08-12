{ pkgs, yosys ? pkgs.yosys, opensta ? pkgs.opensta }:

let
  # nixpkgs marks or-tools broken on Python >= 3.14; pin OpenROAD's stack below that.
  py = pkgs.python313;
  orTools = (pkgs.or-tools.override { python3 = py; }).overrideAttrs (old: {
    # Upstream packaging probe; fails under Nix PYTHONPATH even when the C++ lib is fine.
    # OpenROAD only needs the C++ or-tools. Same pattern as nixpkgs' python_math_opt_.* exclude.
    checkPhase = ''
      runHook preCheck
      ctest --output-on-failure -E 'python_math_opt_.*|python_contrib_check_dependencies'
      runHook postCheck
    '';
  });
  openroad = pkgs.openroad.override {
    python3 = py;
    or-tools = orTools;
  };

  # sky130_fd_sc_hd contains the source views for the HD standard-cell library.
  # Its Liberty files are stored as JSON and are converted with the official
  # SkyWater helper so OpenSTA gets a normal Liberty file.
  skywaterPdk = pkgs.fetchFromGitHub {
    owner = "google";
    repo = "skywater-pdk";
    rev = "7198cf647113f56041e02abf3eb623692820c5e1";
    hash = "sha256-Qh9HjSiW0/je+hZZ2eoZtYE7QEjPRR5x5dHu9XWBfYs=";
  };

  sky130FdScHd = pkgs.stdenvNoCC.mkDerivation {
    pname = "sky130-fd-sc-hd-lib";
    version = "0.0.2";
    src = pkgs.fetchFromGitHub {
      owner = "google";
      repo = "skywater-pdk-libs-sky130_fd_sc_hd";
      rev = "v0.0.2";
      hash = "sha256-4T55Y51YHBwQt18KIapmRv4tYz09644Us0xyn9KCdCc=";
    };
    nativeBuildInputs = [ pkgs.python3 ];
    dontConfigure = true;
    dontFixup = true;

    buildPhase = ''
      runHook preBuild
      mkdir generated
      cp -r ${skywaterPdk}/scripts/python-skywater-pdk skywater-pdk
      chmod -R u+w skywater-pdk

      # The Liberty generator imports dataclasses-json for optional metadata
      # classes.  Liberty generation itself does not need that dependency.
      sed -i \
        -e '/^from dataclasses_json import dataclass_json$/d' \
        -e '/^@dataclass_json$/d' \
        skywater-pdk/skywater_pdk/sizes.py
      sed -i \
        -e '/^import dataclasses_json$/d' \
        -e '/^from dataclasses_json import dataclass_json$/d' \
        skywater-pdk/skywater_pdk/utils.py

      # v0.0.2 uses a space in composite Liberty JSON keys while the helper
      # expects the historical comma separator (for example,
      # `comp_attribute,capacitive_load_unit`).
      find . -type f -name '*.lib.json' -exec sed -i \
        's/"comp_attribute /"comp_attribute,/g' {} +

      export PYTHONPATH="$PWD/skywater-pdk"
      python -c 'from skywater_pdk.liberty import main; raise SystemExit(main())' \
        "$PWD" tt_025C_1v80 --output_directory "$PWD/generated"

      # The pinned SkyWater JSON emits named Liberty declarations as
      # ``declaration name ()`` while Liberty parsers require
      # ``declaration (name)``.  Normalize all such declarations (table
      # templates and operating conditions) in the generated view.
      sed -E -i \
        -e 's/^([[:space:]]*)(ff|latch) ([^ (]+) \(([^)]*)\) \{$/\1\2 (\3, \4) {/' \
        -e 's/^([[:space:]]*)statetable ("[^"]*"|[^ (]+) \(([^)]*)\) \{$/\1statetable (\2, \3) {/' \
        -e 's/^([[:space:]]*)([[:alpha:]_][[:alnum:]_]*) ([^ (]+) \(\) \{$/\1\2 (\3) {/' \
        generated/sky130_fd_sc_hd__tt_025C_1v80.lib
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      libdir="$out/share/sky130_fd_sc_hd"
      mkdir -p "$out/lib" "$libdir/cells"

      install -Dm644 generated/sky130_fd_sc_hd__tt_025C_1v80.lib \
        "$out/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"

      # Keep one physical variant of every cell.  This is enough for the
      # open-source synthesis/STA flow and avoids copying every characterization
      # corner and duplicate drive-strength view into the result.
      for cell in cells/*; do
        test -d "$cell" || continue
        name=$(basename "$cell")
        mkdir -p "$libdir/cells/$name"
        find "$cell" -maxdepth 1 -type f \
          \( -name '*_1.lef' -o -name '*_1.v' -o -name '*_1.cdl' \) \
          -exec cp {} "$libdir/cells/$name/" \;
      done

      find "$libdir/cells" -type f -name '*.lef' -print0 | sort -z |
        xargs -0 cat > "$libdir/sky130_fd_sc_hd_merged.lef"
      find "$libdir/cells" -type f -name '*.v' -print0 | sort -z |
        xargs -0 cat > "$libdir/sky130_fd_sc_hd.v"
      install -Dm644 LICENSE "$out/share/licenses/sky130-fd-sc-hd-lib/LICENSE"
      runHook postInstall
    '';
  };

  # Pre-generated SRAM macros from Sram22.  Each macro includes GDS, LEF,
  # Verilog, SPICE and a typical-corner Liberty model.
  sram22Sky130Macros = pkgs.stdenvNoCC.mkDerivation {
    pname = "sram22-sky130-macros";
    version = "unstable-1f20d16";
    src = pkgs.fetchFromGitHub {
      owner = "rahulk29";
      repo = "sram22_sky130_macros";
      rev = "1f20d16";
      hash = "sha256-NUSoS1RpNy6qQ7+8BL021BZrAXEajWWreUrgI5n6c9E=";
    };
    dontConfigure = true;
    dontBuild = true;
    installPhase = ''
      runHook preInstall
      mkdir -p "$out/share/sram22_sky130_macros"
      cp -r . "$out/share/sram22_sky130_macros/"
      install -Dm644 LICENSE "$out/share/licenses/sram22-sky130-macros/LICENSE"
      runHook postInstall
    '';
  };

  # One stable root for scripts.  The two source packages remain separately
  # addressable above, while this view gives consumers a single SKY130_ROOT.
  sky130Root = pkgs.buildEnv {
    name = "sky130-eda-libraries";
    paths = [ sky130FdScHd sram22Sky130Macros ];
  };
in
{
  inherit yosys opensta sky130FdScHd sram22Sky130Macros sky130Root;

  # Keep names close to the upstream PDK/macro repositories as well as the
  # camelCase names used by the Buckyball Nix overlay.
  sky130_fd_sc_hd = sky130FdScHd;
  sram22_sky130_macros = sram22Sky130Macros;

  # OpenROAD provides placement/routing and reports post-placement area.  The
  # actual power estimate is produced by OpenSTA's Liberty-aware report_power.
  inherit openroad;
  magic = pkgs.magic-vlsi;
  netgen = pkgs.netgen;
  klayout = pkgs.klayout;

  sky130 = sky130Root;
}
