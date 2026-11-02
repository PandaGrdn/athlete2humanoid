import { NQ, type Frames } from "../showcase/data";
import { parseNpy } from "./data";

// An attempt's .npz (a2h_env/attempts.py: numpy's zip of .npy files, deflated) -> its qpos frames and fps. The zip is
// read from its central directory (always carries the sizes); deflated entries go through DecompressionStream.

type Entry = { name: string; method: number; csize: number; usize: number; offset: number };

function entries(buf: ArrayBuffer): Entry[] {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  let eocd = -1;
  for (let i = u8.length - 22; i >= Math.max(0, u8.length - 65557); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("not a zip file");
  const n = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const out: Entry[] = [];
  for (let k = 0; k < n; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("bad zip directory");
    const method = dv.getUint16(p + 10, true);
    const csize = dv.getUint32(p + 20, true);
    const usize = dv.getUint32(p + 24, true);
    const nlen = dv.getUint16(p + 28, true);
    const elen = dv.getUint16(p + 30, true);
    const clen = dv.getUint16(p + 32, true);
    const local = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(u8.subarray(p + 46, p + 46 + nlen));
    // the local header's own name/extra lengths say where the data starts
    const dnlen = dv.getUint16(local + 26, true);
    const delen = dv.getUint16(local + 28, true);
    out.push({ name, method, csize, usize, offset: local + 30 + dnlen + delen });
    p += 46 + nlen + elen + clen;
  }
  return out;
}

async function inflate(data: Uint8Array): Promise<ArrayBuffer> {
  if (typeof DecompressionStream === "undefined") throw new Error("this browser can't inflate the attempt file");
  const ds = new DecompressionStream("deflate-raw");
  return new Response(new Blob([data as BlobPart]).stream().pipeThrough(ds)).arrayBuffer();
}

/** Every array in the npz, by name (without the .npy). */
export async function parseNpz(buf: ArrayBuffer): Promise<Record<string, { shape: number[]; data: Float32Array }>> {
  const out: Record<string, { shape: number[]; data: Float32Array }> = {};
  for (const e of entries(buf)) {
    if (!e.name.endsWith(".npy")) continue;
    const raw = new Uint8Array(buf, e.offset, e.csize);
    const body = e.method === 0 ? raw.slice().buffer : e.method === 8 ? await inflate(raw) : null;
    if (!body) throw new Error(`unsupported zip method ${e.method}`);
    out[e.name.slice(0, -4)] = parseNpy(body);
  }
  return out;
}

export type Attempt = { frames: Frames; fps: number; reward: number | null; fell: boolean };

/** Fetch and decode one attempt. */
export async function loadAttempt(url: string): Promise<Attempt> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const z = await parseNpz(await r.arrayBuffer());
  const q = z.qpos;
  if (!q || q.shape.length !== 2 || q.shape[1] !== NQ) throw new Error("the attempt has no qpos");
  return {
    frames: { n: q.shape[0], data: q.data },
    fps: z.fps?.data[0] || 50,
    reward: z.reward ? z.reward.data[0] : null,
    fell: !!z.fell?.data[0],
  };
}
