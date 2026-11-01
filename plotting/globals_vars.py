from tokenize import String

FONT_SIZE = 100

def format_db_name(db: str) -> str:
    match db:
        case "shopparis":
            return "Paris"
        case "busstop":
            return "Shanghai"
        case "cali":
            return "California"
        case "drink":
            return "Amsterdam"
        case "highway":
            return "Manhattan"
        case "spitz":
            return "Spitz"
        case _:
            return db


def format_y_axis(value, tick_number):
    if value >= 1000:
        # 1500 -> "1.5K", so neighbouring ticks don't both read "1K"
        if value % 1000:
            return f"{value / 1000:.1f}K"
        return f"{int(value / 1000)}K"

    return f"{int(value)}"