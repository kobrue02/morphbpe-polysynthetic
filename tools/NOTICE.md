Third-party morphological analyzer artifacts used as rule-based boundary sources.

- `lang-crk/crk-strict-analyzer.hfstol`, `lang-crk/crk-strict-generator-with-morpheme-boundaries.hfstol`:
  from `giellalt/lang-crk` release `fst-v2021.7.8`
  (https://github.com/giellalt/lang-crk/releases/tag/fst-v2021.7.8).
- `uqailaut/Uqailaut.jar`: the Uqailaut Inuktitut Morphological Analyser,
  National Research Council Canada, 2009 (Benoît Farley), from
  https://www.inuktitutcomputing.ca/Uqailaut/download.php?lang=en.
  "You are granted to use it free of charge for research and education
  purposes only" (per its bundled README.txt) -- used here for a term paper,
  within that grant. `uqailaut/BatchDecompose.java` is a thin wrapper written
  for this project that calls its public `applications.Decompose` API in a
  loop over stdin, avoiding a JVM restart per word.
- `apertium-grn/src/*`: vendored source (`.lexc`/`.twol`/`.spellrelax`,
  `COPYING`, `AUTHORS`) from `apertium/apertium-grn`
  (https://github.com/apertium/apertium-grn, commit `957433b`, GPL-3.0), with
  one local patch: `apertium-grn.grn.lexc`'s `Multichar_Symbols` section
  declares the combining-tilde sequence `g̃` (U+0067 U+0303) explicitly --
  modern `hfst-lexc` treats it as two characters otherwise, which combined
  with the Makefile's own `--Werror` flag turns a warning into a hard
  compile failure. Not an issue with the HFST version this file was
  originally written against (2021).
  `apertium-grn/grn.seg.hfst` is compiled from that (patched) source using
  the real Apertium/lttoolbox/HFST-CLI/VISLCG3 toolchain (MacPorts for
  lttoolbox/apertium-core/VISLCG3, HFST built from source since MacPorts
  has no `hfst` port) -- see `apertium-grn/BUILD.md` for the exact steps.
  A from-scratch Python-only replication of the build graph (via the pip
  `hfst` package's compile_lexc_file/compile_twolc_file/compose_intersect,
  skipping the native toolchain) was tried first and abandoned: it produced
  a transducer with genuine cyclic/infinitely-ambiguous analyses for
  ordinary words that the real `hfst-lexc`/`hfst-twolc` binaries don't
  produce, so the native toolchain turned out to be a correctness
  requirement, not just a convenience.
