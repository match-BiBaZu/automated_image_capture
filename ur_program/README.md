# UR16e pose handshake

The ready-to-copy test program consists of four files in the workspace root:

- `BiBaZu_GUI.urp`
- `BiBaZu_GUI_Init.script`
- `BiBaZu_GUI_Wait.script`
- `BiBaZu_GUI_Ack.script`

All four files must be copied into the same directory on the robot. The `.urp`
retains the original graphical `Switch`, seven `MoveJ` nodes and seven fixed
waypoints. A `BeforeStart` script initializes the handshake once, a script
immediately before the `Switch` waits for a new GUI command and only sets
`angle`, and a script after the `Switch` acknowledges the completed movement.
The desktop application does not upload, start or stop the robot program.

## Register contract

| Register | Direction | Meaning |
| --- | --- | --- |
| input integer 42 | GUI to robot | requested whitelisted angle |
| input integer 43 | GUI to robot | unique command sequence |
| output integer 42 | robot to GUI | acknowledged sequence |
| output integer 43 | robot to GUI | 1 ready, 2 moving, 3 reached, -1 rejected |
| output integer 41 | robot to GUI | last reached whitelisted angle |

The angle is written first. The sequence register is written second and acts as
the commit. On startup, the robot acknowledges the existing sequence without
moving, preventing an old register value from causing an unexpected movement.

## PolyScope commissioning

1. Back up the working `.urp` and installation files on the teach pendant.
2. Copy `BiBaZu_GUI.urp` and all three `BiBaZu_GUI_*.script` files together to
   the controller and load `BiBaZu_GUI.urp` without starting it yet.
3. Verify that input registers 42–43 and output registers 41–43 are not used by
   another RTDE client or URCap.
4. Test every waypoint and every direct transition locally, with the speed
   slider reduced and the workcell clear.
5. Start `BiBaZu` manually. The GUI intentionally has no Dashboard `play`, power,
   brake-release, URScript upload or direct joint-motion command.
6. Connect the UR card. A pose can only be requested while RTDE is connected,
   Robot Mode is `RUNNING`, Safety Mode is `NORMAL` or `REDUCED`, and Dashboard
   reports the program as `PLAYING`.

The graphical movement nodes retain the original programmed acceleration and
speed. Reduce the teach-pendant speed slider for initial commissioning. This
does not replace the robot safety configuration or a collision review.
