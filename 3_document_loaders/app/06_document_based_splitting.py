from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
import datetime

class Task:
    def __init__(self, title, description, priority):
        self.title = title
        self.description = description
        self.priority = priority
        self.completed = False

    def mark_completed(self):
        self.completed = True
        print(f"Task '{self.title}' marked as completed.")

    def __str__(self):
        status = "Done" if self.completed else "Pending"
        return f"[{status}] {self.title} (Priority: {self.priority})"

class DeadlineTask(Task):
    def __init__(self, title, description, priority, due_date):
        super().__init__(title, description, priority)
        self.due_date = due_date

    def is_overdue(self):
        return datetime.date.today() > self.due_date and not self.completed

class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        if not isinstance(task, Task):
            raise ValueError("Only instances of Task can be added.")
        self.tasks.append(task)
        print(f"Added: {task.title}")

    def list_pending_tasks(self):
        print("\n--- Current Pending Tasks ---")
        for task in self.tasks:
            if not task.completed:
                print(task)

def main():
    manager = TaskManager()
    
    # Creating tasks
    t1 = Task("Document Review", "Read the technical requirements", "High")
    t2 = DeadlineTask("Code Deployment", "Push updates to production", "Critical", 
                      datetime.date(2026, 7, 5))

    # Managing tasks
    try:
        manager.add_task(t1)
        manager.add_task(t2)
    except ValueError as e:
        print(f"Error: {e}")

    t1.mark_completed()
    manager.list_pending_tasks()

    if t2.is_overdue():
        print(f"Alert: {t2.title} is past its due date!")
    else:
        print(f"Reminder: {t2.title} is due on {t2.due_date}.")

if __name__ == "__main__":
    main()

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    Language = Language.Python,
    chunk_size = 300,
    chunk_overlap = 0,
)

chunk = splitter.split_text(text)

print(len(chunk))

print(chunk[0])