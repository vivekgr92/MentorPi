# Jeeves Search Protocol — Runtime Flow

```
USER: "Find me a trash can"
         |
         v
==========================================================
  PRE-CYCLE — VOICE PIPELINE (background thread)
==========================================================
  WonderEcho wake word detected
  -> arecord 5s audio capture
  -> OpenAI Whisper STT (Google fallback)
  -> _process_voice_command("find me a trash can")
  -> _extract_search_target() sets:
       _search_target = "trash can"
       _search_start_cycle = 0
       _searched_rooms = []
       _vlm_scan_odom = None
  -> Rule 1: _auto_query_knowledge("trash can")
       Queries knowledge graph BEFORE exploration starts.
       Result cached in _search_knowledge_result
  -> _pending_voice_instruction = "find me a trash can"
  -> self.exploring = True (starts agent loop)

  Spoken immediately: "Understood, Sir. Find me a trash can.
                       Right away."

  Time: ~6 seconds (recording + STT + TTS)
==========================================================


==========================================================
  CYCLE 1 — UNDERSTAND & PLAN
==========================================================

  _prepare_agent_context() builds LLM prompt:
    - Sensor context: LiDAR sectors, odom, IMU, battery
    - Camera frame (base64 for cloud LLM)
    - Identity + knowledge context
    - ** KNOWLEDGE GRAPH LOOKUP: "trash can" not yet found.
        Suggested next room: "kitchen" (ranked by
        likelihood). **
    - ** VOICE INSTRUCTION: "find me a trash can" **
    - ** ACTIVE SEARCH TARGET: "trash can" (turn 1) **
    - SEARCHED ROOMS: [none yet]
    - UNSEARCHED ROOMS (by likelihood): [none known]
    - Auto-VLM: first scan triggers immediately
      -> _run_auto_vlm_scan() calls identify_objects()
      -> Objects at spawn: e.g. [wall, door]
      -> Auto-registered with position (Rule 4)
      -> No known room nearby -> no auto-label yet

  _run_agent_turn() — LLM reasons with 7 tools:
    Round 1 (max 3 rounds per turn):
      Tool 1: speak("Right away, Sir. Searching for a
               trash can.")
      Tool 2: explore_frontier()
        -> Nav2 sends goal to nearest frontier
        -> Returns after NAV2_TOOL_REEVAL_INTERVAL (8s):
           {status: "in_progress", distance_remaining: 2.1}

  Movement limiter: 1 movement tool executed, further
  movement calls rejected this turn.

  LLM stop_reason: end_turn (or tool_use with results)

  Time: ~12 seconds (LLM call + Nav2 drive start)
==========================================================
         |
         v
==========================================================
  CYCLE 2 — NAVIGATING TO FRONTIER
==========================================================

  _prepare_agent_context():
    - Fresh sensor data (LiDAR, odom, camera)
    - ** ACTIVE SEARCH TARGET: "trash can" (turn 2) **
    - Distance moved since last VLM scan: 0.6m
    - Auto-VLM NOT triggered (< 1.0m threshold)
    - No new VLM results to inject

  _run_agent_turn():
    LLM sees: Nav2 was in_progress, no VLM results yet.
    Round 1:
      Tool 1: explore_frontier()
        -> Nav2 still driving (or re-sends goal)
        -> Returns in_progress after 8s

  LLM: "Still moving to unexplored area."

  Time: ~10 seconds
==========================================================
         |
         v
==========================================================
  CYCLE 3 — ARRIVED AT FRONTIER, FIRST SCAN
==========================================================

  _prepare_agent_context():
    - Odom shows robot moved 1.8m total
    - AUTO-VLM TRIGGERED (>= 1.0m since last scan)

    _run_auto_vlm_scan():
      1. Calls identify_objects() via tool registry
      2. VLM/YOLO detects: [desk, monitor, chair, keyboard]
      3. Rule 4: auto-registers ALL objects with position:
           add_object("desk", room="", x=3.2, y=1.1)
           add_object("monitor", room="", x=3.2, y=1.1)
           add_object("chair", room="", x=3.2, y=1.1)
           add_object("keyboard", room="", x=3.2, y=1.1)
         Each marked is_new=True (first time seen)
      4. Rule 5: room_guess="office" from _guess_room()
         No known room within 2.0m -> auto-creates:
           add_room("office", x=3.2, y=1.1,
                    description="Auto-labeled from objects:
                    desk, monitor, chair, keyboard")
      5. Rule 6: mark_room_searched("office", "trash can",
                   scan_x=3.2, scan_y=1.1,
                   new_objects_found=4)
      6. Rule 7: scan position stored in office node:
           scan_positions: [{x:3.2, y:1.1, target:"trash can",
                            new_objects:4}]
      7. _searched_rooms = ["office"]

    Prompt injection:
      - ** AUTO-VLM SCAN RESULTS: desk, monitor, chair,
          keyboard **
      - NEW objects this scan: desk, monitor, chair, keyboard
      - TARGET "trash can" NOT in detections
      - SEARCHED ROOMS: [office]

  _run_agent_turn():
    LLM sees: office found, no trash can here.
    Round 1:
      Tool 1: label_room("office", description="Desk with
               monitor and keyboard",
               connections=[])
        -> Recorded in consciousness + knowledge graph
      Tool 2: speak("I found an office with a desk and
               monitor. No trash can here. Moving on.")

    Movement limiter prevents explore_frontier in same round.
    LLM end_turn. Next cycle will explore.

  Knowledge graph:
    (office) --CONTAINS--> (desk)
    (office) --CONTAINS--> (monitor)
    (office) --CONTAINS--> (chair)
    (office) --CONTAINS--> (keyboard)
    office.searched_for = {"trash can": "2026-03-11T..."}

  Time: ~12 seconds
==========================================================
         |
         v
==========================================================
  CYCLE 4 — EXPLORE NEXT FRONTIER
==========================================================

  _prepare_agent_context():
    - ** ACTIVE SEARCH TARGET: "trash can" (turn 4) **
    - SEARCHED ROOMS: [office]
    - UNSEARCHED ROOMS (by likelihood): [] (no other rooms
      labeled yet, but frontiers exist in Nav2 map)
    - No new VLM (moved only 0.3m so far)

  _run_agent_turn():
    Round 1:
      Tool 1: explore_frontier()
        -> Nav2 picks next frontier, starts driving
        -> Returns in_progress after 8s

  Time: ~10 seconds
==========================================================
         |
         v
==========================================================
  CYCLE 5 — ARRIVED, TARGET FOUND!
==========================================================

  _prepare_agent_context():
    - Robot moved 2.3m since last VLM scan
    - AUTO-VLM TRIGGERED

    _run_auto_vlm_scan():
      1. identify_objects() -> VLM detects:
         [sink, microwave, trash can, counter]
      2. Rule 4: auto-registers with position:
           add_object("sink", room="", x=6.1, y=-0.5)
           add_object("microwave", room="", x=6.1, y=-0.5)
           add_object("trash can", room="", x=6.1, y=-0.5)
           add_object("counter", room="", x=6.1, y=-0.5)
         trash can is_new=True
      3. Rule 5: room_guess="kitchen"
         No known room within 2m -> auto-creates:
           add_room("kitchen", x=6.1, y=-0.5)
      4. Rule 6: mark_room_searched("kitchen", "trash can")
      5. CONTAINS edges created: kitchen -> all 4 objects

    Prompt injection:
      - ** AUTO-VLM SCAN RESULTS: sink, microwave,
          trash can, counter **
      - NEW objects this scan: sink, microwave, trash can,
          counter
      - ** TARGET MATCH DETECTED: "trash can" matches your
          search target! APPROACH IMMEDIATELY. **

  _run_agent_turn():
    LLM sees TARGET MATCH DETECTED. Rule 9: approach NOW.
    Round 1:
      Tool 1: navigate_to(target="approach",
               object_name="trash can")

        _approach_object("trash can") executes:
          Step 1: LiDAR front = 1.2m, depth center = 1.1m
                  -> use min(1.2, 1.1) = 1.1m
                  -> drive forward at 0.10 m/s for 0.3s
          Step 5: distance = 0.6m -> still approaching
          Step 10: distance = 0.38m -> slow to 0.05 m/s
          Step 15: distance = 0.15m -> creeping
          Step 18: distance = 0.09m -> STOP!

        Returns: {reached: True, final_distance_m: 0.09}

      navigate_to returns:
        {success: True, status: "reached",
         approach: {reached: True, final_distance_m: 0.09}}

    Rule 11 — auto-triggered in _run_agent_turn():
      current_room = "kitchen" (nearest room to odom)
      Auto-speaks: "Sir, I found the trash can in the
                    kitchen, 9 centimeters ahead."

    Rule 12: world_knowledge.save() — atomic write to disk

    _search_target cleared to None
    _searched_rooms cleared to []

  Knowledge graph (persisted to disk):
    (office) --CONTAINS--> (desk, monitor, chair, keyboard)
    (kitchen) --CONTAINS--> (sink, microwave, trash can,
                             counter)
    office.searched_for = {"trash can": "..."}
    kitchen.searched_for = {"trash can": "..."}

  LLM may additionally call:
    speak("Found it, Sir! The trash can is right here
           in the kitchen.")

  GOAL COMPLETE

  Time: ~15 seconds (VLM + approach creep)
==========================================================
         |
         v
==========================================================
  NEXT SESSION — INSTANT RECALL (Rule 12)
==========================================================

  USER: "Find me a trash can"

  _extract_search_target("trash can")
  _auto_query_knowledge("trash can"):
    -> get_room_for_object("trash can") = "kitchen"
    -> Returns: {found: True, room: "kitchen",
                 action: "navigate_directly"}

  _prepare_agent_context() injects:
    ** KNOWLEDGE GRAPH LOOKUP: "trash can" was previously
       found in "kitchen". Navigate there directly with
       navigate_to(target="kitchen",
       object_name="trash can"). Do NOT explore. **

  LLM Round 1:
    Tool 1: speak("Right away, Sir. I know where it is.")
    Tool 2: navigate_to(target="kitchen",
             object_name="trash can")
      -> Nav2 plans path to kitchen coordinates
      -> Arrives -> _approach_object() -> stops at 10cm

  Auto-announce: "Sir, I found the trash can in the kitchen,
                  10 centimeters ahead."

  No exploration needed — Jeeves remembered.

  Time: ~15 seconds (Nav2 drive + approach only)
==========================================================
```

## Key Architecture Differences from Naive Flow

1. **Voice is a background thread**, not a tool. WonderEcho wake word triggers
   recording, STT runs, result queued to agent loop.

2. **Rule 1 (knowledge check) is auto-injected** by the system into the first
   prompt. The LLM does NOT need to call `query_knowledge()` — the system
   already did it.

3. **Auto-VLM scan is distance-based** (1.0m), not cycle-based. Triggered
   inside `_prepare_agent_context()`, results cached and injected into the
   prompt. The LLM doesn't call `identify_objects()` during search — the
   system does it automatically.

4. **Room labeling is automatic** (Rule 5). When VLM detects characteristic
   objects and no known room is nearby, the system auto-creates a room node
   in the graph. The LLM may also call `label_room()` explicitly for better
   descriptions.

5. **`mark_room_searched()` is called by the system** (Rule 6), not the LLM.
   Each auto-VLM scan records the scan position and number of new objects
   found (Rule 7).

6. **Approach is synchronous** — `navigate_to(target="approach")` blocks for
   ~5-10 seconds while `_approach_object()` creeps forward using LiDAR+depth.
   It does NOT return `in_progress`.

7. **Announcement is auto-spoken** (Rule 11) by `_run_agent_turn()` when it
   detects `status=="reached"` on a `navigate_to` call during an active search.
   The LLM doesn't need to call `speak()` for this — though it may add extra
   commentary.

8. **Objects are auto-registered with position** (Rule 4) by
   `identify_objects()` in `tool_handlers.py`. Every VLM scan stores
   `add_object(name, room, confidence, x, y)`.

9. **Movement limiter** (Rule 2) allows exactly 1 navigation tool per LLM
   response. Additional movement calls return an error asking the LLM to
   wait for the next turn.

10. **Search summary is likelihood-ranked** (Rule 10) by
    `ROOM_OBJECT_LIKELIHOOD` mapping in `world_knowledge.py`. Kitchen sorts
    first for "trash can", bathroom for "toilet", etc.
