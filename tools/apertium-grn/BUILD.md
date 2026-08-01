# Building grn.seg.hfst

`grn.seg.hfst` in this directory is a pre-built artifact -- normal use of
this project (running `fst_boundaries_guarani.py` or the full
`run_experiment.py` pipeline) never needs to rebuild it. These are the
steps used to produce it, for reproducing on another machine (e.g. the HPC
cluster, if apertium-grn's coverage ever needs to be regenerated against an
updated corpus).

A from-scratch Python-only build (using the pip `hfst` package's
`compile_lexc_file`/`compile_twolc_file`/`compose_intersect` to replicate
apertium-grn's `Makefile.am` build graph without any native toolchain) was
tried first and does not work correctly: it compiles and runs fast, but
produces a transducer with real cyclic/infinitely-ambiguous analyses for
ordinary words. The real `hfst-lexc`/`hfst-twolc` binaries, invoked with
the same flags the real Makefile uses, don't have this problem -- so the
native toolchain below is required, not just faster.

## 1. Install MacPorts

Download the installer for your macOS version from
https://www.macports.org/install.php and run it (needs sudo).

## 2. Install lttoolbox, apertium-core, VISLCG3, and HFST's build deps via MacPorts

```sh
sudo port install autoconf automake libtool gettext flex bison openfst icu \
    lttoolbox apertium vislcg3
```

`lttoolbox`, `apertium`, and `vislcg3` are themselves available as direct
MacPorts ports. HFST is not, so it's built from source (step 3) against the
`openfst`/`icu`/autotools ports just installed.

## 3. Build HFST from source

MacPorts has no `hfst` port. Build it from source, installed to a
user-writable prefix (no further sudo needed):

```sh
git clone --depth 1 https://github.com/hfst/hfst.git
cd hfst
autoreconf -i
CPPFLAGS="-I/opt/local/include" LDFLAGS="-L/opt/local/lib" \
    PKG_CONFIG_PATH="/opt/local/lib/pkgconfig" \
    ./configure --prefix="$HOME/.local/hfst" --without-python \
    --without-foma --without-sfst --enable-all-tools --with-unicode-handler=icu
make -j"$(sysctl -n hw.ncpu)"
make install
```

`--without-python`: this project already has the pip `hfst` package for
Python-side lookups; only the CLI tools (`hfst-lexc`, `hfst-twolc`,
`hfst-compose-intersect`, `hfst-fst2fst`, ...) are needed here.
`--without-foma --without-sfst`: apertium-grn's build only uses the
OpenFST-backed tools.

## 4. Clone and patch apertium-grn

```sh
git clone https://github.com/apertium/apertium-grn.git
```

Modern `hfst-lexc` treats the combining-tilde sequence `g̃` (U+0067 U+0303)
as two characters unless it's declared as a multichar symbol -- this
wasn't an issue with the HFST version apertium-grn's `.lexc` file (last
touched 2021) was originally written against, and `hfst-lexc -A --Werror`
(the real Makefile's own flags) turns that warning into a hard compile
error. Fixed by adding one line to the `Multichar_Symbols` section of
`apertium-grn.grn.lexc` (already applied in this project's vendored copy,
`tools/apertium-grn/src/apertium-grn.grn.lexc`):

```
Multichar_Symbols

g̃                     ! g + combining tilde (U+0067 U+0303)
%<sg%>               ! Sungular
...
```

Copy the patched lexc file (and the other vendored sources) over the fresh
clone before building, or just build directly from
`tools/apertium-grn/src/`.

## 5. Build

```sh
export PATH="$HOME/.local/hfst/bin:/opt/local/bin:/opt/local/sbin:$PATH"
export DYLD_LIBRARY_PATH="$HOME/.local/hfst/lib:/opt/local/lib:$DYLD_LIBRARY_PATH"
cd apertium-grn
./autogen.sh
make
```

This produces `grn.autoseg.hfst` (among other targets) -- the segmentation
transducer, already converted to the optimized-lookup format `.lookup()`
needs (`hfst-fst2fst -w`). Copy it into this directory as `grn.seg.hfst`:

```sh
cp grn.autoseg.hfst /path/to/term-paper-project/tools/apertium-grn/grn.seg.hfst
```

## Verifying

```python
import hfst
istr = hfst.HfstInputStream("grn.seg.hfst")
seg_fst = istr.read()
istr.close()
print(seg_fst.lookup("ajapo", max_number=5, time_cutoff=3.0))
# [('aja>po', 2.0), ('ã>japo', 4.0), ('ã>japo', 4.0)]
```

Always pass `max_number` and `time_cutoff` to `lookup()` -- some paradigms
in this lexicon are recursive (e.g. stacked diminutive/postposition
suffixes), so unbounded lookup on an arbitrary word is not guaranteed to
terminate quickly.
