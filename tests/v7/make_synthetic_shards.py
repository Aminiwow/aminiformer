"""Synthetic training shards (``amini-shards-v7``) with the exact dtypes of ``schema_v1`` (test data).

This generator simulates small games with the structure the data pipeline guarantees:

* rows sorted by (game, side, tick, seq); ENT/OBJ ragged and contiguous per row;
  EVT sorted by (game, D, owner, idx); D = tap + 20; statuses accepted / rejected
  / not_reached; ``card_vis`` / ``el_vis`` set for base and form cards;
* a FIFO hand/queue cycle per owner applied at landing (D), own elixir with the
  x1/x2/x3 income and the cap, ``pending`` consistent with EVT (tap < t <= D);
* objects born at D + 1 of a play (troops with HP -> ENT, spell projectiles and
  areas without HP -> OBJ), LIFE / TOWERHP / BURROW / PRIV consistent with rows
  (``R.tower_hp`` equals the permuted TOWERHP state);
* wait rows with one candidate per 10-tick block, residues from P_TAP, strata
  base / threat with weights 1 / KAPPA (``schema_v1.KAPPA``);
* learnable structure: plays when elixir is high, the cheapest legal card,
  spells on the nearest enemy troop, troops on the busier lane, abilities on the
  healthiest ready unit;
* ``GAME.form_mismatch`` (bit per side) set on 3 of every 8 games, with
  ``forced_deals`` / ``first_forced_deal`` consistent with it (no EVT.forced_deal event is
  written); HAND ``hand_evo`` / evolved / hero bits on evolution and hero deck slots.

Usage: ``python tests/v7/make_synthetic_shards.py OUT_DIR --shards 2 --games 32 --seed 0``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amini.nn import schema_v1 as S  # noqa: E402

try:                                          # optional: the data pipeline's sampling module (not published)
    from amini.v7 import sampling as WS  # noqa: E402
except ImportError:
    WS = None

# tap residue within a 10-tick block (example: uniform over residues 1-9; never residue 0)
P_TAP = np.array([0.0] + [1.0] * 9)
P_TAP = P_TAP / P_TAP.sum()


def last_wait_tick(real_end: int, native_stop: int) -> int:
    """The last tick a wait candidate may sit on: no wait at or after the recording's stop tick."""
    if WS is not None:
        return WS.last_wait_tick(real_end, native_stop)
    return min(int(real_end), int(native_stop) - 1)


CHAMPION_OBJS = (40, 41, 42, 43)            # objects with a bound ability slot
SPELL_PROJ_OBJ, AREA_OBJ, TOWER_OBJ_K, TOWER_OBJ_P = 300, 301, 10, 11
MINER_OBJ = 50


def rate(t: int) -> int:
    return 178 if t < 2400 else (357 if t < 4800 else 537)


def view(x: float, y: float, owner: int) -> Tuple[float, float]:
    return (x, y) if owner == 0 else (S.ARENA_W - x, S.ARENA_H - y)


class Obj:
    __slots__ = ("uid", "obj", "owner", "family", "hp_max", "born", "died", "x0", "y0", "x1", "y1",
                 "tx", "ty", "src", "tgt", "spawn_event", "play_index", "champion", "burrow_exit",
                 "level", "last_ability", "ab_times")

    def ready_at(self, t: int) -> bool:
        last = max([a for a in self.ab_times if a <= t], default=-10000)
        return self.born + 30 <= t and t - last > 200

    def pos(self, t: int) -> Tuple[float, float]:
        span = max(1, (self.died if self.died >= 0 else self.born + 600) - self.born)
        a = min(max((t - self.born) / span, 0.0), 1.0)
        x = self.x0 + (self.x1 - self.x0) * a
        y = self.y0 + (self.y1 - self.y0) * a
        return 2.0 * round(x / 2.0), 2.0 * round(y / 2.0)      # even game units (exact /2)


def simulate_game(rng: np.random.Generator, gi: int, tag: str) -> Dict[str, Any]:
    real_end = int(rng.integers(3000, 5200))
    level = int(rng.choice([11, 16]))
    decks = []
    for o in range(2):
        cards = rng.choice(np.arange(2, 182), size=8, replace=False).astype(np.int64)
        for j in range(8):                          # some evolution / hero forms
            if rng.random() < 0.25:
                cards[j] = 182 + int(rng.integers(0, 62))
        decks.append(cards)
    deck_cost = [rng.integers(1, 8, size=8).astype(np.int64) for _ in range(2)]
    is_spell = [rng.random(8) < 0.3 for _ in range(2)]
    tower_uid = np.arange(1, 7, dtype=np.int64) + gi * 10000
    tower_max = np.array([4824, 3052, 3052, 4824, 3052, 3052], np.int64)
    uid_next = [gi * 10000 + 100]

    # ---------------------------------------------------------------- objects + events
    objs: List[Obj] = []
    events: List[Dict[str, Any]] = []
    elixir = [60000, 60000]
    hand = [list(rng.permutation(8)) for _ in range(2)]
    cyc = [[int(x) for x in h] for h in hand]           # [hand(4) | queue(4)]
    play_index = [0, 0]
    tower_hp = tower_max.copy()
    thp_list: List[Tuple[int, np.ndarray]] = []
    burrow: List[Tuple[int, int, int, int, int, int]] = []
    next_tap = [int(rng.integers(30, 120)), int(rng.integers(30, 120))]
    pending: List[Dict[str, Any]] = []
    elixir_hist = np.zeros((2, real_end + 2), np.int64)
    hand_hist: List[List[List[int]]] = [[], []]
    src_idx = 0

    def new_uid() -> int:
        uid_next[0] += 1
        return uid_next[0]

    for t in range(0, real_end + 1):
        # landing at D == t-1 is visible at t; apply commands with D == t - 1 at step t-1 -> t
        for ev in [e for e in pending if e["D"] == t - 1]:
            pending.remove(ev)
            o = ev["owner"]
            if ev["status"] == 0:
                elixir[o] = max(0, elixir[o] - ev["cost"] * 10000)
                if ev["kind"] == 1:
                    j = ev["deck"]
                    slot = cyc[o].index(j)
                    nxt = cyc[o][4]
                    cyc[o] = cyc[o][:slot] + [nxt] + cyc[o][slot + 1:4] + cyc[o][5:] + [j]
                    ev["play_index"] = play_index[o]
                    play_index[o] += 1
                    wx, wy = ev["raw_x"], ev["raw_y"]
                    if ev["spell"]:
                        p = Obj()
                        p.uid, p.obj, p.owner, p.family, p.hp_max = new_uid(), SPELL_PROJ_OBJ, o, 4, 0
                        kx, ky = view(9000, 3000, o)
                        dist = math.hypot(wx - kx, wy - ky)
                        p.born, p.died = t, t + max(3, int(math.ceil(dist / 600.0)))
                        p.x0, p.y0, p.x1, p.y1, p.tx, p.ty = kx, ky, wx, wy, wx, wy
                        en = [ob for ob in objs if ob.owner != o and ob.family == 5 and ob.born <= t
                              and (ob.died < 0 or ob.died > t)]
                        p.src, p.spawn_event, p.play_index = -1, ev["idx"], ev["play_index"]
                        p.tgt = en[int(rng.integers(len(en)))].uid if en else -1
                        p.champion, p.burrow_exit, p.level, p.last_ability, p.ab_times = False, -1, level, -10000, []
                        objs.append(p)
                        a = Obj()
                        a.uid, a.obj, a.owner, a.family, a.hp_max = new_uid(), AREA_OBJ, o, 3, 0
                        a.born, a.died = p.died, p.died + 40
                        a.x0 = a.x1 = wx
                        a.y0 = a.y1 = wy
                        a.tx = a.ty = None
                        a.src, a.tgt, a.spawn_event, a.play_index = -1, -1, ev["idx"], ev["play_index"]
                        a.champion, a.burrow_exit, a.level, a.last_ability, a.ab_times = False, -1, level, -10000, []
                        objs.append(a)
                    else:
                        for _ in range(int(rng.integers(1, 4))):
                            u = Obj()
                            u.uid, u.owner, u.family = new_uid(), o, 5
                            u.champion = rng.random() < 0.15
                            u.obj = int(rng.choice(CHAMPION_OBJS)) if u.champion else (
                                MINER_OBJ if rng.random() < 0.08 else int(rng.integers(60, 280)))
                            u.hp_max = int(rng.integers(200, 3000))
                            u.born = t
                            u.died = min(t + int(rng.integers(120, 700)), 32000)
                            if u.died > real_end:
                                u.died = -1
                            ex, ey = view(rng.choice([3500, 14500]), 26000, o)
                            u.x0, u.y0 = wx + rng.normal(0, 300), wy + rng.normal(0, 300)
                            u.x1, u.y1 = ex, ey
                            u.tx = u.ty = None
                            u.src, u.tgt, u.spawn_event, u.play_index = -1, -1, ev["idx"], ev["play_index"]
                            u.burrow_exit = t + int(rng.integers(5, 40)) if u.obj == MINER_OBJ else -1
                            if u.burrow_exit >= 0:
                                burrow.append((u.uid, u.burrow_exit, 6, 0, int(u.x0 / 2), int(u.y0 / 2)))
                            u.level, u.last_ability, u.ab_times = level, -10000, []
                            objs.append(u)
                else:
                    ev["caster_obj_ref"].ab_times.append(t)
        for o in range(2):
            elixir[o] = min(S.ELIXIR_MAX, elixir[o] + rate(t))
            elixir_hist[o, t] = elixir[o]
        # tower damage from enemy troops near towers (rows observe at t, list keyed by tick)
        if t % 25 == 0 and t > 0:
            changed = False
            for k in range(6):
                owner_k = 0 if k < 3 else 1
                n_enemy = sum(1 for ob in objs if ob.family == 5 and ob.owner != owner_k
                              and ob.born <= t and (ob.died < 0 or ob.died > t)
                              and rng.random() < 0.03)
                if n_enemy and tower_hp[k] > 0:
                    tower_hp[k] = max(0, tower_hp[k] - int(rng.integers(50, 400)) * n_enemy)
                    changed = True
            if changed:
                thp_list.append((t, tower_hp.copy()))
        # taps
        for o in range(2):
            if t < next_tap[o] or t + 20 > real_end + 20:
                continue
            r = t % 10
            if r == 0:
                continue
            if elixir[o] < 40000 and rng.random() < 0.9:
                next_tap[o] = t + int(rng.integers(5, 25))
                continue
            champs = [ob for ob in objs if ob.owner == o and ob.champion and ob.born + 30 <= t
                      and (ob.died < 0 or ob.died > t + 1)]
            ready = [ob for ob in champs if ob.ready_at(t)]
            u_ab = rng.random()
            if (ready and u_ab < 0.3) or (champs and u_ab > 0.995):
                cast = max(ready, key=lambda ob: ob.hp_max) if (ready and u_ab < 0.3) else champs[0]
                ev = {"owner": o, "kind": 2, "tap": t, "D": t + 20, "deck": -1, "card": -1,
                      "caster_obj_ref": cast, "cost": 1, "spell": False}
                wx, wy = cast.pos(t)
            else:
                hand_now = cyc[o][:4]
                pend_decks = {p["deck"] for p in pending if p["owner"] == o}
                pend_cost = sum(p["cost"] for p in pending if p["owner"] == o) * 10000
                cands = [j for j in hand_now if j not in pend_decks and
                         (elixir[o] + 20 * rate(t) - pend_cost >= int(deck_cost[o][j]) * 10000
                          or rng.random() < 0.03)]
                if not cands:
                    next_tap[o] = t + 10
                    continue
                j = min(cands, key=lambda jj: (deck_cost[o][jj], jj))
                spell = bool(is_spell[o][j])
                enemies = [ob for ob in objs if ob.owner != o and ob.family == 5 and ob.born <= t
                           and (ob.died < 0 or ob.died > t)]
                if spell and enemies:
                    wx, wy = enemies[int(rng.integers(len(enemies)))].pos(t)
                elif spell:
                    wx, wy = view(rng.choice([3500, 14500]), 25500, o)
                else:
                    left = sum(1 for e in enemies if view(*e.pos(t), o)[0] < 9000)
                    lane = 3500 if left >= len(enemies) - left else 14500
                    wx, wy = view(lane + rng.normal(0, 800), 11000 + rng.normal(0, 1500), o)
                wx = float(np.clip(wx, 500, 17500))
                wy = float(np.clip(wy, 500, 31500))
                ev = {"owner": o, "kind": 1, "tap": t, "D": t + 20, "deck": j,
                      "card": int(decks[o][j]), "caster_obj_ref": None,
                      "cost": int(deck_cost[o][j]), "spell": spell}
            ev["raw_x"], ev["raw_y"] = int(round(wx / 1000.0 - 0.5) * 1000 + 500), int(round(wy / 1000.0 - 0.5) * 1000 + 500)
            ev["raw_x"] = int(np.clip(ev["raw_x"], 500, 17500))
            ev["raw_y"] = int(np.clip(ev["raw_y"], 500, 31500))
            ev["idx"] = src_idx
            src_idx += 1
            u = rng.random()
            if ev["D"] > real_end:
                ev["status"] = 2
            elif u < 0.04 or elixir[o] < ev["cost"] * 10000 - 20 * rate(t):
                ev["status"] = 1
            else:
                ev["status"] = 0
            events.append(ev)
            pending.append(ev)
            next_tap[o] = t + int(rng.exponential(55)) + 8
        for o in range(2):
            hand_hist[o].append(list(cyc[o]))
    return {"real_end": real_end, "level": level, "decks": decks, "deck_cost": deck_cost,
            "is_spell": is_spell, "tower_uid": tower_uid, "tower_max": tower_max, "objs": objs,
            "events": events, "elixir": elixir_hist, "hand_hist": hand_hist, "thp": thp_list,
            "burrow": burrow, "tag": tag}


def tower_state(sim: Dict[str, Any], t: int) -> np.ndarray:
    hp = sim["tower_max"].copy()
    for tk, h in sim["thp"]:
        if tk <= t:
            hp = h
        else:
            break
    return hp


def build_shard(rng: np.random.Generator, n_games: int, serial: int) -> Dict[str, np.ndarray]:
    games, evts, rows, ents, objs_out, lifes, thps, burs, privs = [], [], [], [], [], [], [], [], []
    n_rows = n_evt = n_life = n_thp = n_bur = 0
    for gi in range(n_games):
        tag = "".join(rng.choice(list("0289CGJLPQRUVY"), size=12))
        sim = simulate_game(rng, gi, tag)
        real_end = sim["real_end"]
        g = np.zeros((), S.GAME_V7)
        g["replay_id"] = tag.encode()
        g["split"] = int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=8).digest(), "big") % 100 >= 90
        g["population"] = int(rng.integers(1, 6))
        g["seed"] = -1
        g["six_hp_exact"] = bool(rng.random() < 0.9)
        g["outcome_exact"] = True
        g["first_forced_deal"] = [-1, -1]
        # training-only form mask: 3 of every 8 games by game index (no rng draw, so
        # the rest of the synthetic stream is unchanged), owner 0, owner 1 or both
        hfm = int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=8, person=b"v7fm").digest(), "big")
        fm = (0, 1, 0, 2, 0, 3, 0, 0)[gi % 8]
        g["form_mismatch"] = fm
        g["forced_deals"] = bin(fm).count("1")
        g["first_forced_deal"] = [100 + (hfm >> 8) % 2000 if fm >> o & 1 else -1 for o in range(2)]
        g["source_end"] = g["native_stop"] = g["real_end"] = real_end
        for o in range(2):
            g["deck_card"][o] = sim["decks"][o]
            g["deck_level"][o] = sim["level"]
            g["deck_cost"][o] = sim["deck_cost"][o]
            g["deck_el"][o] = (sim["decks"][o] >= 182).astype(np.uint8)
        g["tower_obj"] = [TOWER_OBJ_K, TOWER_OBJ_P, TOWER_OBJ_P] * 2
        g["tower_uid"] = sim["tower_uid"]
        g["tower_max_hp"] = sim["tower_max"]
        final = tower_state(sim, real_end)
        lost0 = int((final[:3] == 0).sum())
        lost1 = int((final[3:] == 0).sum())
        g["result"] = [int(np.sign(lost1 - lost0)), int(np.sign(lost0 - lost1))]
        g["final_crowns"] = [lost1, lost0]
        g["export_revision"] = 1

        # ---- EVT (world frame), sorted by (D, owner, idx)
        evs = sorted(sim["events"], key=lambda e: (e["D"], e["owner"], e["idx"]))
        ev_rec = np.zeros(len(evs), S.EVT_V7)
        for k, e in enumerate(evs):
            r = ev_rec[k]
            r["game"], r["idx"], r["owner"], r["kind"] = gi, e["idx"], e["owner"], e["kind"]
            r["tap"], r["D"], r["status"], r["deck"] = e["tap"], e["D"], e["status"], e["deck"]
            card = e["card"]
            if e["kind"] == 2:
                card = int(sim["decks"][e["owner"]][int(rng.integers(8))])
                r["caster_obj"] = e["caster_obj_ref"].obj
                r["caster_uid"] = e["caster_obj_ref"].uid
            else:
                r["caster_obj"], r["caster_uid"] = -1, -1
            r["card"] = card
            form_play = card >= 182 and rng.random() < 0.5
            base = card if card < 182 else 2 + ((card - 182) * 7) % 180
            r["card_vis"] = card if form_play else base
            r["el_vis"] = (1 if card < 182 + 44 else 2) if form_play else 0
            r["raw_x"], r["raw_y"] = e["raw_x"], e["raw_y"]
            r["gx"], r["gy"] = e["raw_x"] // 1000, e["raw_y"] // 1000
            r["cost"] = e["cost"]
            r["cost_paid"] = e["cost"] if e["status"] == 0 else 0
            r["pd_flags"] = r["el_vis"]
        g["evt_lo"], g["evt_hi"] = n_evt, n_evt + len(ev_rec)
        n_evt += len(ev_rec)
        evts.append(ev_rec)

        # ---- LIFE (world frame)
        lif = np.zeros(len(sim["objs"]) + 6, S.LIFE_V7)
        for k, ob in enumerate(sim["objs"]):
            L = lif[k]
            L["game"], L["uid"], L["obj_first"], L["obj_last"] = gi, ob.uid, ob.obj, ob.obj
            L["owner"], L["family"], L["born"], L["died"] = ob.owner, ob.family, ob.born, ob.died
            L["gone"] = ob.died                      # removed objects: first absent tick; -1 alive at the end
            bx, by = ob.pos(ob.born)
            L["bx"], L["by"] = int(bx / 2), int(by / 2)
            end = ob.died if ob.died >= 0 else real_end
            lx, ly = ob.pos(end)
            L["lx"], L["ly"] = int(lx / 2), int(ly / 2)
            if ob.tx is not None:
                L["btx"], L["bty"] = int(ob.tx / 2), int(ob.ty / 2)
                L["ltx"], L["lty"] = int(ob.tx / 2), int(ob.ty / 2)
            else:
                L["btx"] = L["bty"] = L["ltx"] = L["lty"] = S.INT16_MIN
            L["src_birth"], L["tgt_last"] = -1, -1
            L["play_birth"], L["spawn_event"] = ob.play_index, ob.spawn_event
        for k in range(6):
            L = lif[len(sim["objs"]) + k]
            L["game"], L["uid"], L["obj_first"], L["obj_last"] = gi, sim["tower_uid"][k], TOWER_OBJ_K, TOWER_OBJ_K
            L["owner"], L["family"], L["born"], L["died"] = (0 if k < 3 else 1), 5, 0, -1
            L["gone"] = -1
            L["btx"] = L["bty"] = L["ltx"] = L["lty"] = S.INT16_MIN
            L["play_birth"], L["spawn_event"], L["src_birth"], L["tgt_last"] = -1, -1, -1, -1
        g["life_lo"], g["life_hi"] = n_life, n_life + len(lif)
        n_life += len(lif)
        lifes.append(lif)

        thp = np.zeros(len(sim["thp"]), S.TOWERHP_V7)
        for k, (tk, h) in enumerate(sim["thp"]):
            thp[k]["game"], thp[k]["tick"], thp[k]["hp"] = gi, tk, h
        g["thp_lo"], g["thp_hi"] = n_thp, n_thp + len(thp)
        n_thp += len(thp)
        thps.append(thp)
        bur = np.zeros(len(sim["burrow"]), S.BURROW_V7)
        for k, (uid, tk, a, b, x, y) in enumerate(sim["burrow"]):
            bur[k] = (gi, uid, tk, a, b, x, y)
        g["bur_lo"], g["bur_hi"] = n_bur, n_bur + len(bur)
        n_bur += len(bur)
        burs.append(bur)

        # ---- rows per side
        for side in range(2):
            opp = 1 - side
            taps = sorted([e for e in sim["events"] if e["owner"] == side], key=lambda e: (e["tap"], e["idx"]))
            tap_ticks = {e["tap"] for e in taps}
            row_specs: List[Tuple[int, int, Any, int, float]] = []
            seq_at: Dict[int, int] = {}
            for e in taps:
                q = seq_at.get(e["tap"], 0)
                seq_at[e["tap"]] = q + 1
                row_specs.append((e["tap"], q, e, 0, 1.0))
            # shared end guard: no wait at t >= native_stop; synthetic games set native_stop = real_end
            last_wait = last_wait_tick(real_end, g["native_stop"])
            for k in range(real_end // 10 + 1):
                h = hashlib.sha256(f"il_wait_v7:{tag}:{side}:{k}".encode()).digest()
                u1 = int.from_bytes(h[:8], "big") / 2 ** 64
                u2 = int.from_bytes(h[8:16], "big") / 2 ** 64
                tk = 10 * k + int(np.searchsorted(np.cumsum(P_TAP), u1, side="right"))
                if tk < 1 or tk > last_wait or tk in tap_ticks or u2 >= S.KAPPA["threat"]:
                    continue
                threat = any(ob.owner != side and ob.born <= tk and (ob.died < 0 or ob.died > tk)
                             and (ob.family in (3, 4) or tk - ob.born < 20 or
                                  (ob.burrow_exit >= 0 and tk < ob.burrow_exit))
                             for ob in sim["objs"])
                kap = S.KAPPA["threat" if threat else "base"]
                if u2 < kap:
                    row_specs.append((tk, 0, None, 2 if threat else 1, 1.0 / kap))
            row_specs.sort(key=lambda r: (r[0], r[1]))
            row_lo = n_rows
            perm_self = np.asarray(S.SELF_SLOT[side])
            for (tk, seq, e, stratum, weight) in row_specs:
                r = np.zeros((), S.ROW_V7)
                r["game"], r["side"], r["seq"], r["tick"] = gi, side, seq, tk
                r["stratum"], r["weight"] = stratum, weight
                r["kind"] = 0 if e is None else e["kind"]
                r["threat"] = 1 if stratum == 2 else 0
                r["elixir"] = sim["elixir"][side, tk]
                r["opp_elixir"] = sim["elixir"][opp, tk]
                hp_w = tower_state(sim, tk)
                hp_s = hp_w[perm_self]
                r["tower_hp"] = hp_s
                r["tower_flags"] = 1 | (2 * (hp_s == 0)) | (4 * (hp_s < sim["tower_max"][perm_self]))
                r["crowns"] = [int((hp_s[3:] == 0).sum()), int((hp_s[:3] == 0).sum())]
                r["multiplier"] = 1 if tk < 2400 else (2 if tk < 4800 else 3)
                cyc = sim["hand_hist"][side][tk]
                hand_now, queue_now = cyc[:4], cyc[4:]
                r["hand_deck"] = hand_now
                costs = sim["deck_cost"][side][hand_now]
                r["hand_cost"] = costs
                r["hand_cost_eng"] = costs
                pend = [p for p in sim["events"] if p["owner"] == side and p["tap"] < tk <= p["D"]]
                pend_cost = sum(p["cost"] for p in pend) * 10000
                el = int(r["elixir"])
                flags = np.zeros(4, np.uint8)
                hevo = np.zeros(4, np.uint8)
                for i, j in enumerate(hand_now):
                    c = int(sim["deck_cost"][side][j]) * 10000
                    aff = el // 10000 >= c // 10000
                    legal = el + 20 * rate(tk) - pend_cost >= c and j not in {p["deck"] for p in pend}
                    flags[i] = (1 if aff else 0) | (2 if legal else 0) | 4
                    # own form state of evolution / hero deck slots, a deterministic
                    # function of the tick (no rng draw): hero bit, or counter 1..3 + evolved at 3
                    dc = int(sim["decks"][side][j])
                    if dc >= 182 + 44:
                        flags[i] |= S.HF_HERO
                    elif dc >= 182:
                        n = (tk // 50 + j) % 3
                        hevo[i] = n + 1
                        flags[i] |= S.HF_EVOLVED if n == 2 else 0
                r["hand_flags"] = flags
                r["hand_evo"] = hevo
                r["next_deck"] = queue_now[0]
                r["queue"] = queue_now[:4]
                r["lab_slot"] = r["lab_deck"] = r["lab_gx"] = r["lab_gy"] = r["lab_cost"] = -1
                r["lab_ability_uid"], r["lab_event"], r["lab_status"] = -1, -1, -1
                # snapshot (self frame)
                alive = [ob for ob in sim["objs"] if ob.born <= tk and (ob.died < 0 or ob.died > tk)]
                ent_list = [ob for ob in alive if ob.hp_max > 0]
                obj_list = [ob for ob in alive if ob.hp_max == 0]
                ent_list.sort(key=lambda ob: (0 if (ob.owner == side and ob.champion) else 1, ob.uid))
                ent_list = ent_list[:S.ENT_CAP]
                obj_list = sorted(obj_list, key=lambda ob: ob.uid)[:S.OBJ_CAP]
                er = np.zeros(len(ent_list), S.ENT_V7)
                ready_uids = []
                for k2, ob in enumerate(ent_list):
                    x, y = view(*ob.pos(tk), side)
                    px, py = view(*ob.pos(tk - 1), side)
                    q = er[k2]
                    q["uid"], q["obj"], q["is_self"] = ob.uid, ob.obj, int(ob.owner == side)
                    q["x2"], q["y2"] = int(x / 2), int(y / 2)
                    q["vx"], q["vy"] = int(x - px), int(y - py)
                    frac = 1.0 - 0.5 * (tk - ob.born) / max(1, ((ob.died if ob.died >= 0 else tk + 600) - ob.born))
                    q["max_hp"], q["hp"] = ob.hp_max, max(1, int(ob.hp_max * frac))
                    q["level"] = ob.level
                    q["state"] = 6 if (ob.burrow_exit >= 0 and tk < ob.burrow_exit) else 1
                    q["deploy_left"] = max(0, 20 - (tk - ob.born))
                    q["head_x"], q["head_y"] = 0, 127
                    ab = 0
                    if ob.champion and ob.owner == side:
                        ab |= 1
                        if ob.ready_at(tk):
                            ab |= 2 | 8       # synthetic: no elixir model, AB_LEGAL == AB_READY
                            ready_uids.append(ob.uid)
                    if tk == ob.born:
                        ab |= 4
                    q["ab"] = ab
                    enemies_here = [o2.uid for o2 in ent_list if o2.owner != ob.owner]
                    u_st = rng.random()
                    q["shot_target_uid"] = (enemies_here[0] if (u_st < 0.3 and enemies_here)
                                            else (999999 if u_st > 0.95 else -1))
                    q["born_dt"] = min(32767, tk - ob.born)
                    q["spawn_event"] = ob.spawn_event
                orr = np.zeros(len(obj_list), S.OBJ_V7)
                for k2, ob in enumerate(obj_list):
                    x, y = view(*ob.pos(tk), side)
                    px, py = view(*ob.pos(tk - 1), side)
                    q = orr[k2]
                    q["uid"], q["obj"], q["is_self"], q["family"] = ob.uid, ob.obj, int(ob.owner == side), ob.family
                    q["x2"], q["y2"] = int(x / 2), int(y / 2)
                    q["vx"], q["vy"] = int(x - px), int(y - py)
                    if ob.tx is not None:
                        tx, ty = view(ob.tx, ob.ty, side)
                        q["tx2"], q["ty2"] = int(tx / 2), int(ty / 2)
                        q["eta"] = max(0, ob.died - tk - 1) if ob.died >= 0 else -1
                        q["flags"] = 1 | 2 | (8 if tk == ob.born else 0)
                    else:
                        q["tx2"] = q["ty2"] = S.INT16_MIN
                        q["eta"] = -1
                        q["flags"] = 8 if tk == ob.born else 0
                    q["target_uid"] = ob.tgt
                    q["source_uid"] = int(sim["tower_uid"][0 if ob.owner == 0 else 3]) if ob.family == 4 else -1
                    q["area_left"] = (ob.died - tk) if ob.family == 3 and ob.died >= 0 else -1
                    q["countdown"] = -1
                    q["born_dt"] = min(32767, tk - ob.born)
                    q["spawn_event"] = ob.spawn_event
                if e is not None:
                    r["lab_event"] = e["idx"]
                    r["lab_status"] = e["status"]
                    if e["kind"] == 1:
                        r["lab_slot"] = hand_now.index(e["deck"]) if e["deck"] in hand_now else -1
                        r["lab_deck"] = e["deck"]
                        vx_, vy_ = view(e["raw_x"], e["raw_y"], side)
                        r["lab_gx"], r["lab_gy"] = int(vx_ // 1000), int(vy_ // 1000)
                        r["lab_cost"] = e["cost"]
                    else:
                        cu = e["caster_obj_ref"].uid
                        r["lab_ability_uid"] = cu if cu in ready_uids else -1
                r["n_ent"], r["n_obj"] = len(er), len(orr)
                ents.append(er)
                objs_out.append(orr)
                rows.append(r)
                pr = np.zeros((), S.PRIV_V7)
                ocyc = sim["hand_hist"][opp][tk]
                pr["opp_hand"], pr["opp_queue"] = ocyc[:4], ocyc[4:]
                pr["opp_el"] = (sim["decks"][opp][ocyc[:4]] >= 182).astype(np.uint8)
                pr["opp_evo"] = 0
                privs.append(pr)
                n_rows += 1
            g["row_lo"][side], g["row_hi"][side] = row_lo, n_rows
        games.append(g)
    out = {
        "games": np.array(games, dtype=S.GAME_V7),
        "events": np.concatenate(evts).astype(S.EVT_V7),
        "rows": np.array(rows, dtype=S.ROW_V7),
        "entity": np.concatenate(ents).astype(S.ENT_V7) if ents else np.zeros(0, S.ENT_V7),
        "object": np.concatenate(objs_out).astype(S.OBJ_V7) if objs_out else np.zeros(0, S.OBJ_V7),
        "life": np.concatenate(lifes).astype(S.LIFE_V7),
        "towerhp": np.concatenate(thps).astype(S.TOWERHP_V7),
        "burrow": np.concatenate(burs).astype(S.BURROW_V7),
        "priv": np.array(privs, dtype=S.PRIV_V7),
    }
    return out


def write_shard(root: Path, serial: int, arrays: Dict[str, np.ndarray]) -> Path:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(arrays[name].tobytes())
    sid = f"s{serial:05d}-{digest.hexdigest()[:16]}"
    d = root / "shards" / sid
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arr in arrays.items():
        assert arr.dtype == S.SHARD_TABLES[name], name
        np.save(d / f"{name}.npy", arr, allow_pickle=False)
        files[name] = hashlib.sha256((d / f"{name}.npy").read_bytes()).hexdigest()
    rows = arrays["rows"]
    manifest = {
        "schema": S.SHARD_SCHEMA, "synthetic": True, "shard_id": sid, "ability_legal_rule": 2,
        "dtypes": {k: np.lib.format.dtype_to_descr(v) for k, v in S.SHARD_TABLES.items()},
        "files": files,
        "counts": {"games": int(len(arrays["games"])), "rows": int(len(rows)),
                   "rows_by_kind": {str(k): int((rows["kind"] == k).sum()) for k in range(3)},
                   "rows_by_stratum": {str(k): int((rows["stratum"] == k).sum()) for k in range(3)},
                   "cap_truncations": 0},
    }
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    return d


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.out)
    rng = np.random.default_rng(a.seed)
    paths = []
    for s in range(a.shards):
        arrays = build_shard(rng, a.games, s)
        paths.append(str(write_shard(root, s, arrays)))
    gen = root / "generations" / "synthetic"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "MANIFEST.json").write_text(json.dumps(
        {"schema": S.SHARD_SCHEMA, "generation": "synthetic",
         "shards": [Path(p).name for p in paths], "superseded": []}, indent=1))
    print(json.dumps({"shards": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
