// Every hand-written source file stays at or under 500 lines (CONTRIBUTING.md). Lists the ones over, longest first, and fails when
// there are any. Data (JSON, test fixtures) doesn't count.
//   npm run sizes
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const LIMIT = 500;
const ROOTS = ["src", "test", "tools", "examples"];
const EXT = /\.(ts|js|mjs|py|html)$/;

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const files: string[] = [];
const walk = (p: string): void => {
  const s = statSync(p, { throwIfNoEntry: false });
  if (!s) return;
  if (s.isDirectory()) for (const f of readdirSync(p)) walk(join(p, f));
  else if (EXT.test(p)) files.push(p);
};
for (const r of ROOTS) walk(join(root, r));

const over = files
  .map((f) => ({ f: relative(root, f), n: readFileSync(f, "utf8").split("\n").length }))
  .filter((x) => x.n > LIMIT)
  .sort((a, b) => b.n - a.n);
for (const x of over) console.log(`over ${LIMIT}: ${x.f} (${x.n} lines)`);
console.log(`${files.length} files checked, ${over.length} over the limit`);
process.exit(over.length ? 1 : 0);
