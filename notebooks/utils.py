# Imports
from scipy.stats import shapiro

def hex_to_rgb_normalized(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

def categorize_skewness(skew_value):
    if skew_value > 1:
        return "stark rechtsschief"
    elif skew_value < -1:
        return "stark linksschief"
    elif skew_value > 0.5:
        return "leicht rechtschief"
    elif skew_value < 0.5:
        return "leicht linksschief"
    else:
        return "eher symmetrisch"
    
def shapiro_wilk_test(data,alpha):
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    # interpret
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')  