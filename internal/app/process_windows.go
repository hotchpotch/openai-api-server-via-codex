//go:build windows

package app

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
)

func configureDaemonProcess(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x00000008 | 0x00000200}
}

// configureProcessGroup mirrors the Unix helper: CREATE_NEW_PROCESS_GROUP so the
// Codex CLI's child processes can be terminated as a tree rather than orphaned.
func configureProcessGroup(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x00000200}
}

// killProcessGroup relies on taskkill /T, which already walks the tree.
func killProcessGroup(pid int) error { return forceKillProcess(pid) }
func processAlive(pid int) bool {
	output, err := exec.Command("tasklist", "/FI", "PID eq "+strconv.Itoa(pid), "/FO", "CSV", "/NH").Output()
	return err == nil && !strings.Contains(strings.ToLower(string(output)), "no tasks") && strings.Contains(string(output), fmt.Sprintf("\"%d\"", pid))
}
func terminateProcess(pid int) error {
	return exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T").Run()
}
func forceKillProcess(pid int) error {
	return exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F").Run()
}
