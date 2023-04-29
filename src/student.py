class Student:

    def __init__(self, name: str, rollNo: str, videoUrl: str):
        self.name = name
        self.rollNo = rollNo
        self.videoUrl = videoUrl
    
    def __hash__(self) -> int:
        return hash(self.rollNo)
    
    def __eq__(self, other) -> bool:
        return self.rollNo == other.rollNo

    def __repr__(self) -> str:
        return f'name: {self.name}, rollNo: {self.rollNo}, videoUrl: {self.videoUrl}'
    
    def __str__(self) -> str:
        return f'name: {self.name}, rollNo: {self.rollNo}, videoUrl: {self.videoUrl}'