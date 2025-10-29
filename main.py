from domain.GameMonitor import GameMonitor

SOURCE = 0

if __name__ == '__main__':
    game_monitor = GameMonitor(SOURCE)
    game_monitor.run()