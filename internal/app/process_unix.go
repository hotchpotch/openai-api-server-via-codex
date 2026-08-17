//go:build !windows

package app

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

func configureDaemonProcess(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}

// configureProcessGroup puts the child in its own process group so the whole
// tree can be signalled at once. The Codex CLI is a Node shim that spawns the
// real binary as a grandchild: killing only the direct child leaves that
// grandchild alive, holding the stdout pipe open, so the reader never sees EOF
// and Wait never returns.
func configureProcessGroup(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup signals the whole group led by pid, falling back to the
// single process when the group has already gone.
func killProcessGroup(pid int) error {
	if err := syscall.Kill(-pid, syscall.SIGKILL); err == nil {
		return nil
	}
	return forceKillProcess(pid)
}
func processAlive(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = process.Signal(syscall.Signal(0))
	return err == nil || errors.Is(err, os.ErrPermission)
}
func terminateProcess(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return process.Signal(syscall.SIGTERM)
}
func forceKillProcess(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return process.Kill()
}
