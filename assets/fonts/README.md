# Model label font

Chilanka by The Chilanka Project Authors, licensed under [SIL OFL 1.1](OFL.txt).

`chilanka-latin.woff2` is a Latin subset (U+0020–00FF, en/em dash, ellipsis) of the installed Chilanka-Regular.otf, converted with FontTools. Original font: https://gitlab.com/smc/fonts/fonts-smc-chilanka

The renderer embeds this font into the models SVG; viewers need no installed font or network request. FontTools is only needed to rebuild the subset, not for daily rendering.

`chilanka-metrics.json` stores glyph advances divided by units-per-em from the same subset. It lets the renderer size label boxes to the actual handwriting without a font library at runtime.
