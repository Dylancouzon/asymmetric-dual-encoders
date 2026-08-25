# Windows box setup (Dylan, one-time, host side)

1. PowerShell (admin): `wsl --install` (Ubuntu), reboot when asked.
2. Create `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   memory=20GB
   ```
   Start at 20 GB; raise toward 26 only if the host stays stable under a real encoding load — the 32 GB also carries Windows, the GPU driver, and file cache.
3. Inside WSL (Ubuntu terminal): install git, tmux, the CUDA **toolkit** (never a driver inside WSL — it passes through from Windows), and Claude Code.
4. Clone the repo into the WSL home directory (`~/`), never `/mnt/c` — the Windows mount is slow enough to dominate encoding jobs.
5. Optional remote access for oversight: Tailscale installed inside the WSL distro (simplest supported path), or `networkingMode=mirrored` in `.wslconfig` (Win11 22H2+) plus sshd. Run a connectivity test either way.
6. Launch Claude Code from the WSL terminal in the repo and point it at `instructions-m7.md`. (Commit/push on the M7 work branch is already granted in the mandate; all licensing decisions are settled; progress lands in `m7/STATUS.md` on GitHub.)
