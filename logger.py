import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

class SessionLogger:
    def __init__(self, session_id: str = None, log_dir: str = "logs"):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def log(self, role: str, content: Any, metadata: Dict = None):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "role": role,         
            "content": content,
            "metadata": metadata or {}
        })

    def export_json(self) -> Path:
        """
        Export the conversation to a JSON file,
        Entry includes timestamp, session_id, role, content and metadata.
        Each session will have a unique session id, and all logs from the same session will be stored in the same file.
        """
        path = self.log_dir / f"session_{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, default=str)
        return path

    def export_csv(self) -> Path:
        """
        Export the conversation to a CSV file,
        Entry includes timestamp, session_id, role, content and metadata.
        Each session will have a unique session id, and all logs from the same session will be stored in the same file.
        """
        path = self.log_dir / f"session_{self.session_id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "session_id", "role", "content", "metadata"])
            writer.writeheader()
            for event in self.events:
                writer.writerow({**event, 
                    "content": str(event["content"]),
                    "metadata": json.dumps(event["metadata"])
                })
        return path

        

