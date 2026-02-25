# Custom Skills Directory
# Drop .py files here that subclass `backend.skills.base.Skill`
# TamAGI will auto-discover and register them on startup.
#
# Example:
#
# from backend.skills.base import Skill, SkillResult
#
# class WeatherSkill(Skill):
#     name = "weather"
#     description = "Get current weather for a location"
#     parameters = {
#         "location": {"type": "string", "description": "City name", "required": True}
#     }
#
#     async def execute(self, **kwargs):
#         location = kwargs.get("location", "")
#         return SkillResult(success=True, output=f"Weather for {location}: Sunny, 72°F")
