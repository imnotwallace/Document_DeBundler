use serde::{Deserialize, Serialize};
use std::process::{Child, Command, Stdio};
use std::io::{BufReader, BufRead, Write};

#[derive(Debug, Serialize, Deserialize)]
pub struct PythonCommand {
    pub command: String,
    pub file_path: Option<String>,
    pub options: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PythonEvent {
    pub event_type: String,
    pub data: serde_json::Value,
}

pub struct PythonProcess {
    child: Option<Child>,
}

impl PythonProcess {
    pub fn new() -> Self {
        PythonProcess { child: None }
    }

    /// Starts the Python backend process
    pub fn start(&mut self, python_script_path: &str) -> Result<(), String> {
        // TODO: Detect Python executable (venv or system)
        // For now, we'll use 'python' and assume it's in PATH or venv is activated

        let child = Command::new("python")
            .arg(python_script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python process: {}", e))?;

        self.child = Some(child);
        Ok(())
    }

    /// Sends a command to the Python process via stdin
    pub fn send_command(&mut self, command: PythonCommand) -> Result<(), String> {
        if let Some(ref mut child) = self.child {
            if let Some(ref mut stdin) = child.stdin {
                let command_json = serde_json::to_string(&command)
                    .map_err(|e| format!("Failed to serialize command: {}", e))?;

                writeln!(stdin, "{}", command_json)
                    .map_err(|e| format!("Failed to write to stdin: {}", e))?;

                stdin.flush()
                    .map_err(|e| format!("Failed to flush stdin: {}", e))?;

                Ok(())
            } else {
                Err("Python process stdin not available".to_string())
            }
        } else {
            Err("Python process not started".to_string())
        }
    }

    /// Reads events from Python process stdout (non-blocking approach needed)
    pub fn read_event(&mut self) -> Result<Option<PythonEvent>, String> {
        if let Some(ref mut child) = self.child {
            if let Some(ref mut stdout) = child.stdout {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();

                match reader.read_line(&mut line) {
                    Ok(0) => Ok(None), // EOF
                    Ok(_) => {
                        let event: PythonEvent = serde_json::from_str(&line)
                            .map_err(|e| format!("Failed to parse event: {}", e))?;
                        Ok(Some(event))
                    }
                    Err(e) => Err(format!("Failed to read from stdout: {}", e)),
                }
            } else {
                Err("Python process stdout not available".to_string())
            }
        } else {
            Err("Python process not started".to_string())
        }
    }

    /// Stops the Python process
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(mut child) = self.child.take() {
            child.kill()
                .map_err(|e| format!("Failed to kill Python process: {}", e))?;
            child.wait()
                .map_err(|e| format!("Failed to wait for Python process: {}", e))?;
        }
        Ok(())
    }
}

impl Drop for PythonProcess {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
