from domain.GameMonitor import GameMonitor

# SOURCE = "http://192.168.1.22:8080/video"
SOURCE = 0

if __name__ == '__main__':
    game_monitor = GameMonitor(SOURCE)
    game_monitor.run()