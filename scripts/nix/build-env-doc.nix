{ pkgs }:

{
  # Rust documentation generator
  mdbook = pkgs.mdbook;

  # mdbook plugins
  mdbook-linkcheck = pkgs.mdbook-linkcheck;
  mdbook-pdf = pkgs.mdbook-pdf;
  mdbook-toc = pkgs.mdbook-toc;
  mdbook-mermaid = pkgs.mdbook-mermaid;
}
