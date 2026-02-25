{ pkgs }:

{
  # Node.js environment (for bbdev Motia backend)
  nodejs = pkgs.nodejs_22;
  pnpm = pkgs.nodePackages.pnpm;

  # UV (motia install uses it for Python deps; avoid auto-install via broken pip)
  uv = pkgs.uv;
  # Allure CLI for sardine Allure reports
  allure = pkgs.allure;

  # Build tools (for compiling native modules)
  gcc = pkgs.gcc;
  gnumake = pkgs.gnumake;
  pkg-config = pkgs.pkg-config;
}
