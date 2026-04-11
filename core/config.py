from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class RobotSettings(BaseModel):
    wheel_diameter: float = 10.5
    speed: int = 5


class Settings(BaseSettings):
    
    robot: RobotSettings = RobotSettings()

    # Настройка источника данных
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )


settings = Settings()