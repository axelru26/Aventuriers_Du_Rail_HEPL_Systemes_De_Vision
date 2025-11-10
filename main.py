from domain.GameMonitor import GameMonitor

# SOURCE = "http://192.168.1.72:8080/video"
SOURCE = "http://10.111.38.31:8080/video"

if __name__ == '__main__':
    game_monitor = GameMonitor(SOURCE)
    game_monitor.run()