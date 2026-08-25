# Windows box setup (Dylan, one-time, host side)

1. PowerShell (admin): `wsl --install` (Ubuntu), reboot when asked.
2. Create `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   memory=20GB
   ```
   Start at 20 GB; raise toward 26 only if the host stays stable under a real encoding load — the 32 GB also carries Windows, the GPU driver, and file cache.
3. Inside WSL (Ubuntu terminal):
   ```bash
   sudo apt update
   sudo apt install -y git tmux gh python3-venv build-essential
   gh auth login          # once — gives the session git push + private clone
   nvidia-smi             # sanity: GPU already visible through the Windows driver

   # CUDA toolkit — the wsl-ubuntu repo ships toolkit only, never a driver
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install -y cuda-toolkit-12-6
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   ```
   The toolkit block is optional at first: PyTorch wheels bundle their own CUDA runtime, so training runs on the Windows driver alone — the toolkit only becomes necessary if the session compiles a CUDA extension. Then install Claude Code.
4. Clone the repo into the WSL home directory (`~/`), never `/mnt/c` — the Windows mount is slow enough to dominate encoding jobs.
5. Optional remote access for oversight: Tailscale installed inside the WSL distro (simplest supported path), or `networkingMode=mirrored` in `.wslconfig` (Win11 22H2+) plus sshd. Run a connectivity test either way.
6. Launch Claude Code from the WSL terminal in the repo and point it at `instructions-m7.md`. (Commit/push on the M7 work branch is already granted in the mandate; all licensing decisions are settled; progress lands in `m7/STATUS.md` on GitHub.)
