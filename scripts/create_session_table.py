from ecommerce_agent.application.services.session import SessionService

class CreateSessionTable:
  def __init__(self):
    self.session_service = SessionService()
  
  def create_session_table(self):
    self.session_service.create_table()
    
create_session_table = CreateSessionTable()
create_session_table.create_session_table()