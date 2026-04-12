from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from math import pi, sin, cos, radians

class RobotSettings(BaseModel):
    wheel_radius: float = 0.040 # m - 40 mm
    theta: float = radians(30)
    wheel_mounting_radius: float = 0.125
    
class DataPath(BaseModel):
    train_val_dir: str = "data/raw/Data_Set_(A+B).xlsx"
    test_dir: str = "data/raw/Data_Set_C.xlsx"
    report_dir_raw: str = "reports/raw/"
    report_dir_scaled: str = "reports/scaled/"
    report_dir_balanced: str = "reports/balanced/"
    


class Settings(BaseSettings):
    
    robot: RobotSettings = RobotSettings()
    data_path: DataPath = DataPath() 

    # Настройка источника данных
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )