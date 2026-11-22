// Push-optimal Sokoban solver, in two parts.
//
// First the level is cut into rooms. The generator runs a one-square hallway
// across the board on the row the player starts in, and every room opens onto
// that hallway alone; a box pushed out of its room can never come back, so no
// optimal answer ever moves one. That makes the rooms independent subproblems,
// and each is searched on its own instead of the whole board at once — which is
// what lets four rooms of four boxes be solved inside the wall.
//
// Inside a room the search is push-optimal: a state is the set of boxes plus the
// region the player can walk, canonicalised to that region's lowest cell, so
// walking is free and only pushes are counted. The heuristic is each box's
// walking distance to its nearest goal, which never overestimates, so the first
// goal state popped is optimal.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const In = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const Out = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const FLOOR = 0;
const WALL = 1;

type Grid = { w: number; h: number; wall: Uint8Array; goal: Uint8Array; boxes: number[]; player: number };
type Push = { box: number; to: number; stand: number };

/** Read one level: walls, goals, box cells and the player's cell, as flat arrays. */
function readLevel(text: string): Grid {
  const lines = text.split('\n');
  while (lines.length && lines[lines.length - 1].trim() === '') lines.pop();
  const h = lines.length;
  const w = lines.reduce((widest, line) => Math.max(widest, line.length), 0);
  const wall = new Uint8Array(w * h).fill(WALL);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // A space is open floor in XSB; only a '#' is wall, and past a short
      // row's end the board is wall.
      const ch = lines[y][x] ?? '#';
      if (ch === '#') continue;
      const at = y * w + x;
      wall[at] = FLOOR;
      if (ch === '.' || ch === '*' || ch === '+') goal[at] = 1;
      if (ch === '$' || ch === '*') boxes.push(at);
      if (ch === '@' || ch === '+') player = at;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

/** One step from `at` towards `d`, or -1 when it would leave the board. */
function step(at: number, d: number, w: number, h: number): number {
  const x = at % w;
  if (d === -1) return x === 0 ? -1 : at - 1;
  if (d === 1) return x === w - 1 ? -1 : at + 1;
  const next = at + d;
  return next < 0 || next >= w * h ? -1 : next;
}

const DIRECTIONS = (w: number): number[] => [-w, w, -1, 1];

/** The cells a box can still be pushed from on its way to a goal; outside them it is dead. */
function pushableCells(grid: Grid): Uint8Array {
  const { w, h, wall, goal } = grid;
  const live = new Uint8Array(w * h);
  const queue: number[] = [];
  for (let at = 0; at < w * h; at++) if (goal[at]) { live[at] = 1; queue.push(at); }
  for (let head = 0; head < queue.length; head++) {
    const at = queue[head];
    for (const d of DIRECTIONS(w)) {
      // A box arriving at `at` came from behind it, pushed by a player standing two back.
      const cameFrom = step(at, -d, w, h);
      const stoodAt = step(at, -2 * d, w, h);
      if (cameFrom < 0 || stoodAt < 0) continue;
      if (wall[cameFrom] || wall[stoodAt]) continue;
      if (live[cameFrom]) continue;
      live[cameFrom] = 1;
      queue.push(cameFrom);
    }
  }
  return live;
}

/** Walking distance from every floor cell to its nearest goal, boxes ignored. */
function goalDistances(grid: Grid): Int32Array {
  const { w, h, wall, goal } = grid;
  const dist = new Int32Array(w * h).fill(-1);
  const queue: number[] = [];
  for (let at = 0; at < w * h; at++) if (goal[at]) { dist[at] = 0; queue.push(at); }
  for (let head = 0; head < queue.length; head++) {
    const at = queue[head];
    for (const d of DIRECTIONS(w)) {
      const next = step(at, d, w, h);
      if (next < 0 || wall[next] || dist[next] >= 0) continue;
      dist[next] = dist[at] + 1;
      queue.push(next);
    }
  }
  return dist;
}

/** Where the player can walk with these boxes standing, and the lowest cell of it. */
function walkRegion(grid: Grid, boxes: number[], from: number): { seen: Uint8Array; lowest: number } {
  const { w, h, wall } = grid;
  const blocked = new Uint8Array(w * h);
  for (const cell of boxes) blocked[cell] = 1;
  const seen = new Uint8Array(w * h);
  const queue = [from];
  seen[from] = 1;
  let lowest = from;
  for (let head = 0; head < queue.length; head++) {
    const at = queue[head];
    if (at < lowest) lowest = at;
    for (const d of DIRECTIONS(w)) {
      const next = step(at, d, w, h);
      if (next < 0 || wall[next] || blocked[next] || seen[next]) continue;
      seen[next] = 1;
      queue.push(next);
    }
  }
  return { seen, lowest };
}

/** The shortest walk from one cell to another around the standing boxes, as u/d/l/r steps. */
function walkTo(grid: Grid, boxes: number[], from: number, to: number): string {
  if (from === to) return '';
  const { w, h, wall } = grid;
  const blocked = new Uint8Array(w * h);
  for (const cell of boxes) blocked[cell] = 1;
  const cameFrom = new Int32Array(w * h).fill(-1);
  const cameBy = new Array<string>(w * h).fill('');
  const queue = [from];
  cameFrom[from] = from;
  for (let head = 0; head < queue.length && cameFrom[to] < 0; head++) {
    const at = queue[head];
    for (const [d, ch] of [[-w, 'u'], [w, 'd'], [-1, 'l'], [1, 'r']] as [number, string][]) {
      const next = step(at, d, w, h);
      if (next < 0 || wall[next] || blocked[next] || cameFrom[next] >= 0) continue;
      cameFrom[next] = at;
      cameBy[next] = ch;
      queue.push(next);
    }
  }
  if (cameFrom[to] < 0) return '';
  let back = '';
  for (let at = to; at !== from; at = cameFrom[at]) back = cameBy[at] + back;
  return back;
}

/** The direction that takes a cell to its neighbour `to`, as the letter to write. */
function letterFor(from: number, to: number, w: number): string {
  const delta = to - from;
  if (delta === -w) return 'u';
  if (delta === w) return 'd';
  if (delta === -1) return 'l';
  return 'r';
}

class Frontier {
  private keys: number[] = [];
  private states: string[] = [];

  get size(): number {
    return this.keys.length;
  }

  push(key: number, state: string): void {
    this.keys.push(key);
    this.states.push(state);
    let child = this.keys.length - 1;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (this.keys[parent] <= this.keys[child]) break;
      this.exchange(parent, child);
      child = parent;
    }
  }

  pop(): { key: number; state: string } {
    const key = this.keys[0];
    const state = this.states[0];
    const lastKey = this.keys.pop() as number;
    const lastState = this.states.pop() as string;
    if (this.keys.length) {
      this.keys[0] = lastKey;
      this.states[0] = lastState;
      let parent = 0;
      for (;;) {
        const left = 2 * parent + 1;
        const right = left + 1;
        let least = parent;
        if (left < this.keys.length && this.keys[left] < this.keys[least]) least = left;
        if (right < this.keys.length && this.keys[right] < this.keys[least]) least = right;
        if (least === parent) break;
        this.exchange(least, parent);
        parent = least;
      }
    }
    return { key, state };
  }

  private exchange(a: number, b: number): void {
    const key = this.keys[a];
    this.keys[a] = this.keys[b];
    this.keys[b] = key;
    const state = this.states[a];
    this.states[a] = this.states[b];
    this.states[b] = state;
  }
}

/** A state's name: its boxes sorted, then the lowest cell the player can reach beside them. */
function identify(boxes: number[], lowest: number): string {
  return `${boxes.join(',')}@${lowest}`;
}

/**
 * Search one room push-optimally. `boxes` are every box standing on the board at
 * this moment (the other rooms' included, since the player walks around them);
 * the room's own boxes are those on `allowed` cells. Returns the moves and where
 * the boxes end up, or null when the search runs out of budget.
 */
function searchRoom(
  grid: Grid,
  boxes: number[],
  player: number,
  allowed: Set<number>,
  live: Uint8Array,
  distance: Int32Array,
): { moves: string; boxes: number[]; player: number } | null {
  const { w, h, wall, goal } = grid;
  // Every box in this room is pushed freely, including one that already stands
  // on a goal: a push-optimal line often lifts a box off a goal to let another
  // through, then puts it back. Only the final state must have them all home.
  const mine = (box: number): boolean => allowed.has(box);

  if (boxes.every((box) => !mine(box) || goal[box] === 1)) return { moves: '', boxes, player };

  const first = walkRegion(grid, boxes, player);
  const firstKey = identify(boxes, first.lowest);
  const cost = new Map<string, number>([[firstKey, 0]]);
  const parent = new Map<string, { previous: string; push: Push } | null>([[firstKey, null]]);
  const held = new Map<string, number[]>([[firstKey, boxes]]);

  const estimate = (state: number[]): number => {
    let total = 0;
    for (const box of state) {
      if (!mine(box) || goal[box] === 1) continue;
      const d = distance[box];
      if (d < 0) return 1 << 20;
      total += d;
    }
    return total;
  };

  const open = new Frontier();
  open.push(estimate(boxes), firstKey);
  const budget = 1200000;
  let expanded = 0;

  while (open.size) {
    const { state } = open.pop();
    const boxesHere = held.get(state) as number[];
    if (boxesHere.every((box) => !mine(box) || goal[box] === 1)) {
      const walked = assemble(grid, parent, state, boxes, player);
      return { moves: walked.moves, boxes: boxesHere, player: walked.player };
    }
    if (++expanded > budget) return null;

    const spent = cost.get(state) as number;
    const lowest = Number(state.slice(state.indexOf('@') + 1));
    const region = walkRegion(grid, boxesHere, lowest).seen;
    const taken = new Uint8Array(w * h);
    for (const box of boxesHere) taken[box] = 1;

    for (let index = 0; index < boxesHere.length; index++) {
      const box = boxesHere[index];
      if (!mine(box)) continue;
      for (const d of DIRECTIONS(w)) {
        const stand = step(box, -d, w, h);
        const landing = step(box, d, w, h);
        if (stand < 0 || landing < 0) continue;
        if (wall[stand] || wall[landing]) continue;
        if (!region[stand]) continue;
        if (taken[landing]) continue;
        if (!allowed.has(landing)) continue;
        if (!goal[landing] && !live[landing]) continue;

        const moved = boxesHere.slice();
        moved[index] = landing;
        moved.sort((a, b) => a - b);
        const nextRegion = walkRegion(grid, moved, box);
        const key = identify(moved, nextRegion.lowest);
        const nextCost = spent + 1;
        const seen = cost.get(key);
        if (seen !== undefined && seen <= nextCost) continue;
        cost.set(key, nextCost);
        held.set(key, moved);
        parent.set(key, { previous: state, push: { box, to: landing, stand } });
        open.push(nextCost + estimate(moved), key);
      }
    }
  }
  return null;
}

/** Turn a chain of pushes into the player's moves, filling in the walks between them. */
function assemble(
  grid: Grid,
  parent: Map<string, { previous: string; push: Push } | null>,
  finalState: string,
  startBoxes: number[],
  startPlayer: number,
): { moves: string; player: number } {
  const chain: Push[] = [];
  for (let at = finalState; ;) {
    const link = parent.get(at);
    if (!link) break;
    chain.push(link.push);
    at = link.previous;
  }
  chain.reverse();

  const boxes = startBoxes.slice();
  let player = startPlayer;
  let moves = '';
  for (const { box, to, stand } of chain) {
    moves += walkTo(grid, boxes, player, stand);
    const at = boxes.indexOf(box);
    if (at < 0) throw new Error(`a push names a box at ${box} that is not standing`);
    boxes[at] = to;
    boxes.sort((a, b) => a - b);
    // The player follows the box into the cell it just left.
    player = box;
    moves += letterFor(box, to, grid.w);
  }
  return { moves, player };
}

/**
 * Cut the board into rooms by flood-filling the floor while treating the row the
 * player starts in as a corridor: every room lies wholly above or below it.
 */
function roomsOf(grid: Grid): Int32Array[] {
  const { w, h, wall } = grid;
  const corridor = Math.floor(grid.player / w);
  const label = new Int32Array(w * h).fill(-1);
  const found: Int32Array[] = [];
  let next = 0;
  for (let start = 0; start < w * h; start++) {
    if (wall[start] || label[start] >= 0) continue;
    if (Math.floor(start / w) === corridor) continue;
    const cells: number[] = [start];
    label[start] = next;
    for (let head = 0; head < cells.length; head++) {
      const at = cells[head];
      for (const d of DIRECTIONS(w)) {
        const n = step(at, d, w, h);
        if (n < 0 || wall[n] || label[n] >= 0) continue;
        if (Math.floor(n / w) === corridor) continue;
        label[n] = next;
        cells.push(n);
      }
    }
    found.push(Int32Array.from(cells));
    next++;
  }
  return found;
}

/** Solve a whole level: each room on its own, then the answers joined in order. */
function solveLevel(grid: Grid): string {
  const live = pushableCells(grid);
  const distance = goalDistances(grid);
  const rooms = roomsOf(grid);
  const roomOf = new Map<number, number>();
  rooms.forEach((cells, index) => {
    for (const cell of cells) roomOf.set(cell, index);
  });

  // Order the rooms so each is finished before the next begins. A box in the
  // corridor should not happen, but a level is still attempted if one does.
  const order = [...new Set(grid.boxes.map((box) => roomOf.get(box)).filter((r): r is number => r !== undefined))].sort(
    (a, b) => a - b,
  );

  let moves = '';
  let boxes = grid.boxes.slice();
  let player = grid.player;
  for (const room of order) {
    const allowed = new Set<number>(rooms[room]);
    const answer = searchRoom(grid, boxes, player, allowed, live, distance);
    if (answer === null) return '';
    moves += answer.moves;
    boxes = answer.boxes;
    // The player ends on the cell of the last box it pushed; carrying that
    // forward keeps the walks between rooms honest, and any small error is
    // repaired by the walk search at the start of the next room.
    player = answer.player;
  }
  return moves;
}

const levels = readdirSync(In).sort();
for (const id of levels) {
  const text = readFileSync(join(In, id, 'level.txt'), 'utf8');
  const answer = solveLevel(readLevel(text));
  mkdirSync(join(Out, id), { recursive: true });
  writeFileSync(join(Out, id, 'solution.txt'), `${answer}\n`);
}
