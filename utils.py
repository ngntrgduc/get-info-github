def format_stars(number: int):
    """Format number of stars"""
    if number > 1000:
        return f'{number/1000:.1f}K'
    return number  

if __name__ == '__main__':
    print(format(1000))