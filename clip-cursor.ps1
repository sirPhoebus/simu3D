# Confine the Windows pointer to the simu3d window while that window is in front,
# and hide it. The game runs under WSLg, so its own grab cannot call ClipCursor
# or change the Windows pointer. Alt-Tab shows the pointer and lets it out.
# Closing the game does too.
param(
    [Parameter(Mandatory = $true)][string]$StopFile,
    [Parameter(Mandatory = $true)][string]$LogFile
)

$source = @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class SimuClip {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr value);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern IntPtr GetAncestor(IntPtr hWnd, uint gaFlags);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr hWnd, ref POINT pt);
    [DllImport("user32.dll")] public static extern bool ClipCursor(IntPtr lpRect);
    [DllImport("user32.dll")] public static extern int GetSystemMetrics(int index);
    [DllImport("user32.dll")] public static extern IntPtr CreateCursor(IntPtr inst, int hotX, int hotY, int width, int height, byte[] andPlane, byte[] xorPlane);
    [DllImport("user32.dll")] public static extern bool SetSystemCursor(IntPtr cursor, uint id);
    [DllImport("user32.dll")] public static extern IntPtr LoadCursor(IntPtr inst, IntPtr name);
    [DllImport("user32.dll")] public static extern IntPtr CopyIcon(IntPtr icon);
    [DllImport("user32.dll", EntryPoint = "SystemParametersInfoW")] public static extern bool SystemParametersInfo(uint action, uint param, IntPtr value, uint winIni);

    const uint SPI_SETCURSORS = 0x0057;
    static readonly uint[] CursorIds = {
        32512, 32513, 32514, 32515, 32516,
        32642, 32643, 32644, 32645, 32646, 32648, 32649, 32650
    };
    static IntPtr[] saved;

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT { public int X, Y; }

    public static void MakeDpiAware() {
        if (!SetProcessDpiAwarenessContext(new IntPtr(-4))) {
            SetProcessDPIAware();
        }
    }

    public static IntPtr Find() {
        IntPtr found = IntPtr.Zero;
        EnumProc proc = (h, l) => {
            if (!IsWindowVisible(h)) return true;
            var sb = new StringBuilder(512);
            if (GetWindowText(h, sb, sb.Capacity) <= 0) return true;
            if (!sb.ToString().StartsWith("simu3d")) return true;
            found = h;
            return false;
        };
        EnumWindows(proc, IntPtr.Zero);
        GC.KeepAlive(proc);
        return found;
    }

    public static bool IsGameFocus(IntPtr hwnd) {
        IntPtr fg = GetForegroundWindow();
        if (fg == IntPtr.Zero) return false;
        if (fg == hwnd) return true;
        if (GetAncestor(fg, 2) == hwnd || GetAncestor(hwnd, 2) == fg) return true;
        var sb = new StringBuilder(512);
        if (GetWindowText(fg, sb, sb.Capacity) > 0 && sb.ToString().StartsWith("simu3d")) return true;
        return false;
    }

    public static bool ClipToClient(IntPtr hwnd) {
        RECT client;
        if (!GetClientRect(hwnd, out client)) return false;
        POINT origin = new POINT();
        if (!ClientToScreen(hwnd, ref origin)) return false;
        RECT screen = new RECT {
            Left = origin.X + client.Left,
            Top = origin.Y + client.Top,
            Right = origin.X + client.Right,
            Bottom = origin.Y + client.Bottom
        };
        if (screen.Right <= screen.Left || screen.Bottom <= screen.Top) return false;
        IntPtr mem = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(RECT)));
        try {
            Marshal.StructureToPtr(screen, mem, false);
            return ClipCursor(mem);
        } finally {
            Marshal.FreeHGlobal(mem);
        }
    }

    public static void Release() {
        ClipCursor(IntPtr.Zero);
    }

    public static void CaptureCursors() {
        if (saved != null) return;
        saved = new IntPtr[CursorIds.Length];
        for (int i = 0; i < CursorIds.Length; i++) {
            IntPtr current = LoadCursor(IntPtr.Zero, new IntPtr((int)CursorIds[i]));
            if (current != IntPtr.Zero) saved[i] = CopyIcon(current);
        }
    }

    static IntPtr BlankCursor() {
        int width = GetSystemMetrics(13);
        int height = GetSystemMetrics(14);
        if (width <= 0) width = 32;
        if (height <= 0) height = 32;
        int stride = ((width + 15) / 16) * 2;
        byte[] andPlane = new byte[stride * height];
        byte[] xorPlane = new byte[stride * height];
        for (int i = 0; i < andPlane.Length; i++) andPlane[i] = 0xFF;
        return CreateCursor(IntPtr.Zero, 0, 0, width, height, andPlane, xorPlane);
    }

    public static bool HideCursor() {
        CaptureCursors();
        foreach (uint id in CursorIds) {
            IntPtr blank = BlankCursor();
            if (blank == IntPtr.Zero || !SetSystemCursor(blank, id)) {
                RestoreCursors();
                return false;
            }
        }
        return true;
    }

    public static void RestoreCursors() {
        if (saved != null) {
            for (int i = 0; i < CursorIds.Length; i++) {
                if (saved[i] == IntPtr.Zero) continue;
                IntPtr copy = CopyIcon(saved[i]);
                if (copy != IntPtr.Zero) SetSystemCursor(copy, CursorIds[i]);
            }
        }
        SystemParametersInfo(SPI_SETCURSORS, 0, IntPtr.Zero, 0);
    }
}
"@

Add-Type -TypeDefinition $source -Language CSharp
[SimuClip]::MakeDpiAware()
# A previous run can die while the pointer is blank. Reload the real cursors
# before remembering them.
[SimuClip]::RestoreCursors()
[SimuClip]::CaptureCursors()

$sawWindow = $false
$loggedClip = $false
$loggedHide = $false
$cursorHidden = $false
$missing = 0
$deadline = (Get-Date).AddSeconds(45)

try {
    while ($true) {
        if (Test-Path -LiteralPath $StopFile) { break }
        $hwnd = [SimuClip]::Find()
        if ($hwnd -eq [IntPtr]::Zero) {
            [SimuClip]::Release() | Out-Null
            if ($cursorHidden) {
                [SimuClip]::RestoreCursors()
                $cursorHidden = $false
            }
            if ($sawWindow) {
                $missing += 1
                if ($missing -ge 20) { break }
            } elseif ((Get-Date) -gt $deadline) {
                Add-Content -LiteralPath $LogFile -Value "NO_WINDOW"
                break
            }
            Start-Sleep -Milliseconds 50
            continue
        }
        $sawWindow = $true
        $missing = 0
        if ([SimuClip]::IsGameFocus($hwnd)) {
            $ok = [SimuClip]::ClipToClient($hwnd)
            if (-not $loggedClip) {
                if ($ok) {
                    Add-Content -LiteralPath $LogFile -Value "CLIP_OK"
                } else {
                    Add-Content -LiteralPath $LogFile -Value "CLIP_FAIL"
                }
                $loggedClip = $true
            }
            if (-not $cursorHidden) {
                if ([SimuClip]::HideCursor()) {
                    $cursorHidden = $true
                    if (-not $loggedHide) {
                        Add-Content -LiteralPath $LogFile -Value "HIDE_OK"
                        $loggedHide = $true
                    }
                } elseif (-not $loggedHide) {
                    Add-Content -LiteralPath $LogFile -Value "HIDE_FAIL"
                    $loggedHide = $true
                }
            }
        } else {
            [SimuClip]::Release() | Out-Null
            if ($cursorHidden) {
                [SimuClip]::RestoreCursors()
                $cursorHidden = $false
            }
            $loggedClip = $false
        }
        Start-Sleep -Milliseconds 30
    }
} finally {
    [SimuClip]::Release() | Out-Null
    [SimuClip]::RestoreCursors()
    Add-Content -LiteralPath $LogFile -Value "RELEASED"
}
