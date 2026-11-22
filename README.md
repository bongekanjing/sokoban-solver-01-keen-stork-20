# Push-optimal Sokoban solver

A program that answers every instance of a set in one run: it reads each level in XSB,
works out its push-optimal line, and writes the moves in LURD.

    MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts

`main.ts` is the whole entry: Node 22 runs it directly, and it needs nothing installed.
