from abc import ABC, abstractmethod # abstract base class (abc) for creating abstract classes
from datetime import datetime, timezone
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO) # Set up basic logging

class BaseAgent(ABC): # Base class for all agents
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name) # create a logger for the agent
        self.is_active = True
        self.last_update = None
        
    @abstractmethod # forces any class inheriting from BaseAgent to implement a process() method.
    def process(self) -> Dict[str, Any]:
        """Main processing method for the agent"""
        pass
    
    def log_action(self, message: str, data: Dict = None):
        """Log agent actions"""
        self.logger.info(f"{self.name}: {message}")
        if data:
            self.logger.info(f"Data: {data}")
    
    def publish_event(self, event_type: str, data: Dict):
        """Publish event to other agents"""
        event = {
            "type": event_type,
            "source": self.name,
            "timestamp": datetime.now(timezone.utc),
            "data": data
        }
        # For now, just log - we'll add real event system later
        self.log_action(f"Published event: {event_type}", data)
        return event