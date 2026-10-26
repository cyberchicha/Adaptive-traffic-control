import os
import sys
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from core.rl.env import TrafficEnv

# Текстова розшифровка фаз для вашої зручності
PHASE_NAMES = {
    0: "Північ-Південь: ЗЕЛЕНИЙ",
    1: "Північ-Південь: ОЧИЩЕННЯ (Зелений закінчується)",
    2: "Північ-Південь: ЖОВТИЙ",
    3: "Захід-Схід: ЗЕЛЕНИЙ",
    4: "Захід-Схід: ОЧИЩЕННЯ (Зелений закінчується)",
    5: "Захід-Схід: ЖОВТИЙ"
}

def main():
    sumo_cfg_path = os.path.join(ROOT_DIR, "data", "sumo", "cross.sumocfg")
    
    if not os.path.exists(sumo_cfg_path):
        logging.error(f"SUMO config not found at: {sumo_cfg_path}")
        return

    logging.info("--- Ініціалізація RL Середовища ---")
    env = TrafficEnv(sumo_cfg_path=sumo_cfg_path, render_mode="human")
    obs, info = env.reset()
    
    logging.info("Запуск 60 кроків (перемикання світлофора кожні 10 секунд)...")
    
    for step in range(60):
        # Тримаємо світло 9 секунд (дія 0), на 10-ту секунду - перемикаємо (дія 1)
        if step > 0 and step % 10 == 0:
            action = 1
            logging.info(">>> АГЕНТ ВІДПРАВИВ КОМАНДУ ПЕРЕМКНУТИ СВІТЛОФОР <<<")
        else:
            action = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_phase = int(obs[-1])
        phase_name = PHASE_NAMES.get(current_phase, "Невідомо")
        
        # Виводимо статус кожної секунди
        logging.info(f"Секунда {step+1:02d} | {phase_name} | Загальна черга (Штраф): {reward}")
        
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    main()