{ pkgs }:

{
  # Build tool for Scala, Java and more
  mill = pkgs.mill;

  # Scala formatter
  scalafmt = pkgs.scalafmt;

  # Coursier - Scala dependency manager and launcher
  coursier = pkgs.coursier;
}
