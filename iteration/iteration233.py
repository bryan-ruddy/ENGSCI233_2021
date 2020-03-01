# Supplementary classes and functions for ENGSCI233 notebook Iteration.ipynb
# author: David Dempsey

# imports 
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import Checkbox, FloatText,IntText, BoundedFloatText, fixed
from scipy.optimize import root
import traitlets

TEXTSIZE = 12


# EULER METHOD
# ------------
# demonstration of Euler's method
    # ODE to be solved
def dydx(x,y): return (1.+x*y)**2
    # plotting 
def plot_euler_elements(ax, ax2, step, h):
    # initialise ODE
    x = [0,]
    y = [1,]
    h0 = 0.1
    
    # setup axes limits
    xlim = np.array([-0.05,1.15])
    ylim = [-0.9,10]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    for i in range(int(step)):
        y.append(y[-1]+h0*dydx(x[-1], y[-1]))
        x.append(x[-1]+h0)        
        
    if abs(step-int(step))>0.25:
        # plot derivative
        dydx0 = dydx(x[-1], y[-1])
        ys = dydx0*(xlim - x[-1])+y[-1]
        ax.plot(xlim, ys, 'r--')        
        ax.text(0.95*xlim[-1], np.min([1.05*ys[-1],9.]), 'compute derivative: $f^{'+'{:d}'.format(int(step))+'}=(x^{'+'{:d}'.format(int(step))+'},y^{'+'{:d}'.format(int(step))+'})$', ha = 'right', va = 'bottom', color = 'r', size = TEXTSIZE)
    else:    
        dy = 0.4
        dx = 0.04
        ax.arrow(x[-2], y[-2]-dy, h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.02, color= 'r', linewidth = 0.5)
        ax.arrow(x[-1], y[-2]-dy, -h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.02, color= 'r', linewidth = 0.5)
        ax.text(0.5*(x[-1]+x[-2]), y[-2]-2*dy, '$x^{'+'{:d}'.format(int(step))+'}=x^{'+'{:d}'.format(int(step-1))+'}+h$', ha = 'center', va = 'top', color = 'r', size = TEXTSIZE)
        
        ax.arrow(x[-1]+dx, y[-2], 0, y[-1]-y[-2], length_includes_head = True, head_width = 0.02, head_length = 0.2, color= 'r', linewidth = 0.5)
        ax.arrow(x[-1]+dx, y[-1], 0, -y[-1]+y[-2], length_includes_head = True, head_width = 0.02, head_length = 0.2, color= 'r', linewidth = 0.5)
        
        ax.text(x[-1]+2*dx, 0.5*(y[-1]+y[-2]), '$y^{'+'{:d}'.format(int(step))+'}=y^{'+'{:d}'.format(int(step-1))+'}+hf^{'+'{:d}'.format(int(step-1))+'}$', ha = 'left', va = 'center', color = 'r', size = TEXTSIZE)
                
    ax.plot(x,y,'ko-', mfc = 'k')
    
    ax.plot(x[-1],y[-1],'ko', mfc = 'w')
    
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
    
    # second plot, effect of step size
    x = [0,]
    y = [1,]
    x0 = [0,]
    y0 = [1,]
    
    while x[-1] < 1.:
        y.append(y[-1]+h*dydx(x[-1], y[-1]))
        x.append(x[-1]+h)    
    while x0[-1] < 1.:
        y0.append(y0[-1]+h0*dydx(x0[-1], y0[-1]))
        x0.append(x0[-1]+h0)    

    y0 = y0[:-1]
    x0 = x0[:-1]
    
    ax2.plot(x,y,'ko-', mfc = 'k', label = 'h={:3.2f}'.format(h))
    ax2.plot(x0,y0,'ko-', mfc = 'k', alpha = 0.5, label = 'h={:3.2f}'.format(h0))
    
    ax2.set_xlabel('$x$', size = TEXTSIZE)
    ax2.set_ylabel('$y(x)$', size = TEXTSIZE)
    ax2.set_ylim([0,20])
    ax2.set_xlim(xlim)
    
    ax2.legend(loc=2,prop={'size':TEXTSIZE})
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    for t in ax2.get_xticklabels()+ax2.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    

# EULER TRADEOFFS
# ------------------
# an exercise to show the tradeoffs in using Euler's method
#    # analytic function (true solution)
#def varsin(x, *p): 
#    return np.sin((p[1]*x+p[0])*x)
#    # ODE to solve
#def dvarsin(x, *p): 
#    return (2*p[1]*x+p[0])*np.cos((p[1]*x+p[0])*x)
    # ODE to solve
def dvarsin(x, *p): 
    return np.sin(p[0]*np.sin(x)*np.sqrt(x)+np.cos(p[1]*x)/(x+1))
    # plotting
def plot_tradeoffs_elements(ax, steps, predict_value, p, show_analytic=False):
        
    x = np.linspace(0,10., 1001)
    
    ax.set_xlim([0,10])
    ax.plot([0,10],[0,0],'k:')
    
    xs = np.linspace(0, predict_value,10*steps)
    h = xs[1]-xs[0]
    ya = 0.*xs
    for i in range(len(xs)-1):
        ya[i+1] = ya[i] + h/2*(dvarsin(xs[i], *p)+dvarsin(xs[i+1], *p))
    
    if show_analytic:
        ax.plot(xs,ya, 'k-', label = 'analytical solution')
    
    ax.set_xlabel('$x$',size = TEXTSIZE)
    ax.set_ylabel('$y(x)$',size = TEXTSIZE)
    
    # plot Euler steps
    h = predict_value/steps
    xs = np.arange(steps+1)*h
    ys = 0.*xs
    for i in range(steps):
        ys[i+1] = ys[i] + h*dvarsin(xs[i], *p)
        
    ax.plot(xs,ys, '.b-', label = 'Euler')
    
    # plot error bar
    xest = xs[-1]
    yest = ys[-1]
    ytrue = ya[-1]
    
    ax.plot([xest, xest], [yest, ytrue], 'r-', lw = 2, label = 'error')
    ymid = 0.5*(yest+ytrue)
    err = abs((yest-ytrue)/ytrue)*100
    if err < 1.0:
        wgt = 'bold'
        err_str = ' err < 1%'
    else:
        wgt = 'normal'
        err_str = ' err = {:d}%'.format(int(err))
    
    ax.text(xest, ymid, err_str, color = 'r', fontweight = wgt, size = TEXTSIZE)
    ax.text(0.3, 1.02, 'solving $dy/dx = sin['+'{:2.1f}'.format(p[0])+'sin(x)\sqrt{x}+cos('+'{:2.1f}'.format(p[1])+'x)/(x+1)]$', transform=ax.transAxes, ha='left', va = 'bottom', size = 15)    
        
    ax.legend(loc = 4, prop = {'size':TEXTSIZE})
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    # box setup
def tradeoffs_setup(lock=True):
    box1 = IntText(value = 20, description='Euler steps')
    if lock:
        box2 = BoundedFloatText(value = 2.2, description='predict value', max=6)
    else:
        box2 = BoundedFloatText(value = 2.2, description='predict value')
    return box1, box2
    
    
# IMPROVED EULER METHOD
# ---------------------
# demonstration of improved Euler method
    # hard-coded implementation
class ImprovedEuler(object):
    ''' This class is a hard coded Improved Euler method for a particular function and time step.
    
        It is hard coded because you will be implementing Improved Euler in the lab and I don't want
        the more enterprising students visiting the source code for inspiration...
        
        If you're reading this, that means you...
    '''
    def __init__(self):
        self.x = [0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7, 0.7999999999999999, 0.8999999999999999]
        self.y = [1, 1.111605, 1.2510911476245623, 1.4303787320668615, 1.669257368774675, 2.0030917028294266, 2.501624425724359, 3.3228472650308203, 4.90801761586217, 9.010617182240603]
        self.dydx0s = [1.0, 1.23467765676025, 1.5630456214364068, 2.042365737783545, 2.781233121133108, 4.0061857953154485, 6.254874227126293, 11.062230004937321, 24.269555812652037,-1]
        self.dydx1s = [1.2321000000000002, 1.555045295730995, 2.0227060674095765, 2.735206996372725, 3.8954535599619247, 5.9644686625831955, 10.169582559002931, 20.641177011689674, 57.78243551491661,-1]
        self.yps = [1.1, 1.235072765676025, 1.407395709768203, 1.634615305845216, 1.9473806808879859, 2.4037102823609713, 3.1271118484369884, 4.429070265524553, 7.334973197127374,-1]
        self.xs = [[0, 0.3333333333333333, 0.6666666666666666, 1.0], [0, 0.25, 0.5, 0.75, 1.0], [0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0], [0, 0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333333, 0.9999999999999999, 1.1666666666666665], [0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714, 0.7142857142857142, 0.857142857142857, 0.9999999999999998, 1.1428571428571426], [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], [0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 0.6666666666666667, 0.7777777777777779, 0.8888888888888891, 1.0000000000000002], [0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 0.9999999999999999, 1.0999999999999999], [0, 0.09090909090909091, 0.18181818181818182, 0.2727272727272727, 0.36363636363636365, 0.4545454545454546, 0.5454545454545455, 0.6363636363636365, 0.7272727272727274, 0.8181818181818183, 0.9090909090909093, 1.0000000000000002], [0, 0.08333333333333333, 0.16666666666666666, 0.25, 0.3333333333333333, 0.41666666666666663, 0.49999999999999994, 0.5833333333333333, 0.6666666666666666, 0.75, 0.8333333333333334, 0.9166666666666667, 1.0], [0, 0.07692307692307693, 0.15384615384615385, 0.23076923076923078, 0.3076923076923077, 0.38461538461538464, 0.46153846153846156, 0.5384615384615385, 0.6153846153846154, 0.6923076923076923, 0.7692307692307692, 0.846153846153846, 0.9230769230769229, 0.9999999999999998, 1.0769230769230766], [0, 0.07142857142857142, 0.14285714285714285, 0.21428571428571427, 0.2857142857142857, 0.3571428571428571, 0.4285714285714285, 0.4999999999999999, 0.5714285714285713, 0.6428571428571427, 0.7142857142857141, 0.7857142857142855, 0.8571428571428569, 0.9285714285714283, 0.9999999999999997, 1.0714285714285712], [0, 0.06666666666666667, 0.13333333333333333, 0.2, 0.26666666666666666, 0.3333333333333333, 0.39999999999999997, 0.4666666666666666, 0.5333333333333333, 0.6, 0.6666666666666666, 0.7333333333333333, 0.7999999999999999, 0.8666666666666666, 0.9333333333333332, 0.9999999999999999, 1.0666666666666667], [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0], [0, 0.058823529411764705, 0.11764705882352941, 0.1764705882352941, 0.23529411764705882, 0.29411764705882354, 0.35294117647058826, 0.411764705882353, 0.4705882352941177, 0.5294117647058824, 0.5882352941176471, 0.6470588235294118, 0.7058823529411765, 0.7647058823529412, 0.823529411764706, 0.8823529411764707, 0.9411764705882354, 1.0], [0, 0.05555555555555555, 0.1111111111111111, 0.16666666666666666, 0.2222222222222222, 0.2777777777777778, 0.33333333333333337, 0.38888888888888895, 0.44444444444444453, 0.5000000000000001, 0.5555555555555557, 0.6111111111111113, 0.6666666666666669, 0.7222222222222224, 0.777777777777778, 0.8333333333333336, 0.8888888888888892, 0.9444444444444448, 1.0000000000000002], [0, 0.05263157894736842, 0.10526315789473684, 0.15789473684210525, 0.21052631578947367, 0.2631578947368421, 0.3157894736842105, 0.3684210526315789, 0.42105263157894735, 0.47368421052631576, 0.5263157894736842, 0.5789473684210527, 0.631578947368421, 0.6842105263157894, 0.7368421052631577, 0.7894736842105261, 0.8421052631578945, 0.8947368421052628, 0.9473684210526312, 0.9999999999999996, 1.052631578947368], [0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35, 0.39999999999999997, 0.44999999999999996, 0.49999999999999994, 0.5499999999999999, 0.6, 0.65, 0.7000000000000001, 0.7500000000000001, 0.8000000000000002, 0.8500000000000002, 0.9000000000000002, 0.9500000000000003, 1.0000000000000002], [0, 0.047619047619047616, 0.09523809523809523, 0.14285714285714285, 0.19047619047619047, 0.23809523809523808, 0.2857142857142857, 0.3333333333333333, 0.38095238095238093, 0.42857142857142855, 0.47619047619047616, 0.5238095238095237, 0.5714285714285714, 0.6190476190476191, 0.6666666666666667, 0.7142857142857144, 0.7619047619047621, 0.8095238095238098, 0.8571428571428574, 0.9047619047619051, 0.9523809523809528, 1.0000000000000004], [0, 0.045454545454545456, 0.09090909090909091, 0.13636363636363635, 0.18181818181818182, 0.2272727272727273, 0.27272727272727276, 0.31818181818181823, 0.3636363636363637, 0.40909090909090917, 0.45454545454545464, 0.5000000000000001, 0.5454545454545455, 0.5909090909090909, 0.6363636363636364, 0.6818181818181818, 0.7272727272727272, 0.7727272727272726, 0.818181818181818, 0.8636363636363634, 0.9090909090909088, 0.9545454545454543, 0.9999999999999997, 1.0454545454545452], [0, 0.043478260869565216, 0.08695652173913043, 0.13043478260869565, 0.17391304347826086, 0.21739130434782608, 0.2608695652173913, 0.30434782608695654, 0.34782608695652173, 0.3913043478260869, 0.4347826086956521, 0.4782608695652173, 0.5217391304347825, 0.5652173913043477, 0.6086956521739129, 0.652173913043478, 0.6956521739130432, 0.7391304347826084, 0.7826086956521736, 0.8260869565217388, 0.869565217391304, 0.9130434782608692, 0.9565217391304344, 0.9999999999999996, 1.0434782608695647], [0, 0.041666666666666664, 0.08333333333333333, 0.125, 0.16666666666666666, 0.20833333333333331, 0.24999999999999997, 0.29166666666666663, 0.3333333333333333, 0.375, 0.4166666666666667, 0.45833333333333337, 0.5, 0.5416666666666666, 0.5833333333333333, 0.6249999999999999, 0.6666666666666665, 0.7083333333333331, 0.7499999999999998, 0.7916666666666664, 0.833333333333333, 0.8749999999999997, 0.9166666666666663, 0.9583333333333329, 0.9999999999999996, 1.0416666666666663], [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24000000000000002, 0.28, 0.32, 0.36, 0.39999999999999997, 0.43999999999999995, 0.4799999999999999, 0.5199999999999999, 0.5599999999999999, 0.6, 0.64, 0.68, 0.7200000000000001, 0.7600000000000001, 0.8000000000000002, 0.8400000000000002, 0.8800000000000002, 0.9200000000000003, 0.9600000000000003, 1.0000000000000002], [0, 0.038461538461538464, 0.07692307692307693, 0.11538461538461539, 0.15384615384615385, 0.19230769230769232, 0.23076923076923078, 0.2692307692307693, 0.3076923076923077, 0.34615384615384615, 0.3846153846153846, 0.423076923076923, 0.46153846153846145, 0.4999999999999999, 0.5384615384615383, 0.5769230769230768, 0.6153846153846152, 0.6538461538461536, 0.6923076923076921, 0.7307692307692305, 0.7692307692307689, 0.8076923076923074, 0.8461538461538458, 0.8846153846153842, 0.9230769230769227, 0.9615384615384611, 0.9999999999999996, 1.038461538461538], [0, 0.037037037037037035, 0.07407407407407407, 0.1111111111111111, 0.14814814814814814, 0.18518518518518517, 0.2222222222222222, 0.25925925925925924, 0.2962962962962963, 0.3333333333333333, 0.37037037037037035, 0.4074074074074074, 0.4444444444444444, 0.48148148148148145, 0.5185185185185185, 0.5555555555555556, 0.5925925925925926, 0.6296296296296295, 0.6666666666666665, 0.7037037037037035, 0.7407407407407405, 0.7777777777777775, 0.8148148148148144, 0.8518518518518514, 0.8888888888888884, 0.9259259259259254, 0.9629629629629624, 0.9999999999999993, 1.0370370370370363], [0, 0.03571428571428571, 0.07142857142857142, 0.10714285714285714, 0.14285714285714285, 0.17857142857142855, 0.21428571428571425, 0.24999999999999994, 0.28571428571428564, 0.32142857142857134, 0.35714285714285704, 0.39285714285714274, 0.42857142857142844, 0.46428571428571414, 0.49999999999999983, 0.5357142857142856, 0.5714285714285713, 0.607142857142857, 0.6428571428571427, 0.6785714285714284, 0.7142857142857141, 0.7499999999999998, 0.7857142857142855, 0.8214285714285712, 0.8571428571428569, 0.8928571428571426, 0.9285714285714283, 0.964285714285714, 0.9999999999999997, 1.0357142857142854]]
        self.ys = [[1, 3.2222222222222214, 10.382716049382712, 33.45541838134429], [1, 1.625, 2.640625, 4.291015625, 6.972900390625], [1, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 0.7222222222222221, 0.5216049382716048, 0.3767146776406035, 0.27207171162932475, 0.19649623617673456, 0.14191394834986384, 0.10249340714156832], [1, 0.5918367346938774, 0.3502707205331111, 0.20730307949918814, 0.12268957766278484, 0.07261219902491346, 0.042974566769846734, 0.025433927271950107, 0.01505273246707251], [1, 0.53125, 0.2822265625, 0.149932861328125, 0.0796518325805664, 0.0423150360584259, 0.02247986290603876, 0.011942427168833092, 0.00634441443344258], [1, 0.5061728395061729, 0.25621094345374185, 0.12968702076053598, 0.0656440475454565, 0.033227233942761926, 0.016818723353743688, 0.008513180956833224, 0.004309140978150151, 0.0021811701247426685], [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125], [1, 0.5041322314049588, 0.25414930674134284, 0.12812485711753646, 0.06459187011710517, 0.0325628436127555, 0.016415979011389137, 0.008275824129708573, 0.004172109685224984, 0.002103294965278711, 0.0010603387841487716, 0.0005345509572981411], [1, 0.5138888888888888, 0.2640817901234568, 0.13570869770233196, 0.06973919187480948, 0.03583819582455487, 0.01841685063206292, 0.009464214908143446, 0.0048635548833514935, 0.0024993268150556285, 0.0012843762799591425, 0.0006600266994234482, 0.0003391803872037164], [1, 0.5266272189349113, 0.27733622772311894, 0.14605280631572534, 0.07691538320768967, 0.04050573435197859, 0.021331422232698783, 0.011233707566332494, 0.005915976173985751, 0.003115514079791313, 0.0016407145153930585, 0.000864044922307587, 0.00045502957446967595, 0.00023963095933610153, 0.00012619618568587593], [1, 0.5408163265306123, 0.2924822990420658, 0.158179202543158, 0.0855458952529324, 0.04626461682046344, 0.02502066011718941, 0.013531581491949376, 0.007318100194625683, 0.0039577480644404205, 0.002140414769544309, 0.0011575712529168203, 0.000626033432699913, 0.0003385691013581162, 0.00018310369767326695, 9.902546914982804e-05], [1, 0.5555555555555556, 0.308641975308642, 0.1714677640603567, 0.09525986892242039, 0.05292214940134466, 0.029401194111858146, 0.01633399672881008, 0.009074442627116711, 0.00504135701506484, 0.0028007538972582447, 0.0015559743873656915, 0.000864430215203162, 0.00048023900844620113, 0.0002667994491367784, 0.00014822191618709912, 8.234550899283284e-05], [1, 0.5703125, 0.32525634765625, 0.18549776077270508, 0.10579169169068336, 0.060334324167342857, 0.03440941925168772, 0.019624121916978154, 0.011191882030776604, 0.006382870220677282, 0.0036402306727300123, 0.0020760690555413353, 0.0011840081332384178, 0.0006752546384875351, 0.0003851061610124224, 0.00021963085745239713, 0.00012525822339082025], [1, 0.5847750865051904, 0.3419619017971529, 0.1999708007049095, 0.11693794228072563, 0.06838239531295028, 0.03998832113456262, 0.023384173950661186, 0.013674482344850313, 0.007996496596123538, 0.004676151988736602, 0.002734497183724864, 0.0015990658271609068, 0.000935093857405513, 0.0005468195913547809, 0.00031976647383722486, 0.00018699146739962284, 0.00010934795152434693], [1, 0.5987654320987654, 0.35852004267642124, 0.21466940826921518, 0.1285366210007029, 0.07696328541400113, 0.046082954846655005, 0.02759288037114528, 0.01652166293827835, 0.009892600648228396, 0.005923347301717003, 0.003546695606583638, 0.002123638727398845, 0.0012715614602326418, 0.0007613670471763349, 0.00045588026898829933, 0.00027296534624608043, 0.00016344221349302347, 9.786354758532887e-05], [1, 0.6121883656509696, 0.3747745950384052, 0.22943264682406525, 0.1404559970862006, 0.0859855273020785, 0.05263933942869625, 0.0322251911738002, 0.01972788711747879, 0.012077182972196157, 0.007393510905416484, 0.004526221357609537, 0.0027709000554894947, 0.0016963127763522945, 0.0010384629461879698, 0.0006357349338159039, 0.0003891895301199855, 0.0002382573023726227, 0.00014585834854390474, 8.929278401164252e-05, 5.466400350851246e-05], [1, 0.625, 0.390625, 0.244140625, 0.152587890625, 0.095367431640625, 0.059604644775390625, 0.03725290298461914, 0.023283064365386963, 0.014551915228366852, 0.009094947017729282, 0.0056843418860808015, 0.003552713678800501, 0.002220446049250313, 0.0013877787807814457, 0.0008673617379884035, 0.0005421010862427522, 0.00033881317890172014, 0.00021175823681357508, 0.00013234889800848443, 8.271806125530277e-05], [1, 0.6371882086167802, 0.4060088132002613, 0.2587040283656994, 0.16484315639628466, 0.10503611552688434, 0.06692777429264059, 0.042645588608235835, 0.027173266210689953, 0.01731448481905641, 0.011032585564976988, 0.007029833432559034, 0.004479326971766641, 0.002854174328948812, 0.0018186462277428937, 0.0011588199319631592, 0.0007383863965570244, 0.00047049110528916976, 0.00029979138454933496, 0.000191023535279735, 0.00012171794424853863, 7.755723885224343e-05], [1, 0.6487603305785123, 0.4208899665323406, 0.2730567138247003, 0.17714836392759484, 0.11492683114310906, 0.07455996896474432, 0.048371550113491146, 0.031381542842223595, 0.02035910010838473, 0.013208176516596705, 0.008568940963246622, 0.005559188972023635, 0.0036065812752384737, 0.002339806860381985, 0.0015179738722312878, 0.0009848012311583147, 0.0006388999722803943, 0.000414492957223231, 0.0002689065879506085, 0.0001744559268935766, 0.00011318008480285755, 7.342674923160593e-05, 4.763636210480219e-05], [1, 0.6597353497164462, 0.43525073166548156, 0.2871502936696655, 0.189443199415337, 0.1249823754176798, 0.08245529115457514, 0.05439867034583502, 0.035888725804719135, 0.023677061069654025, 0.01562059416504585, 0.010305458154255202, 0.006798875039385757, 0.004485458201787579, 0.0029592153353948302, 0.0019522989641829786, 0.0012880006398863129, 0.000849739552590403, 0.0005606032208961261, 0.00036984976199007185, 0.0002440029620690644, 0.00016097737951248294, 0.00010620246776910501, 7.006552221440009e-05, 4.622470180118266e-05], [1, 0.6701388888888888, 0.4490861304012346, 0.300950080442494, 0.20167835251875466, 0.1351525070698599, 0.09057095091834362, 0.06069511641402889, 0.04067415787467908, 0.02725733496462869, 0.018266200167268532, 0.012240891084315371, 0.00820309715025301, 0.0054972144097181624, 0.0036838971565125184, 0.0024687227472462363, 0.001654387118814318, 0.0011086691455943173, 0.0007429623093739696, 0.0004978879364901949, 0.0003336540685507209, 0.00022359456677183728, 0.0001498394145380715, 0.00010041321877030487, 6.729080285648903e-05, 4.509418385868883e-05], [1, 0.6799999999999999, 0.4623999999999999, 0.31443199999999993, 0.21381375999999996, 0.14539335679999998, 0.098867482624, 0.06722988818431999, 0.04571632396533759, 0.03108710029642956, 0.021139228201572102, 0.01437467517706903, 0.009774779120406939, 0.006646849801876718, 0.004519857865276168, 0.0030735033483877943, 0.0020899822769037003, 0.0014211879482945162, 0.0009664078048402711, 0.0006571573072913844, 0.0004468669689581414, 0.0003038695388915361, 0.00020663128644624455, 0.0001405092747834463, 9.554630685274347e-05, 6.497148865986557e-05], [1, 0.6893491124260355, 0.47520219880256287, 0.3275802139674472, 0.22581712974679052, 0.15566683796154496, 0.1073087965829585, 0.07397322367996843, 0.05099337608707883, 0.03515223854523482, 0.0242321644409459, 0.016704421049527796, 0.01151517782408277, 0.007937977612459426, 0.0054720378216066455, 0.003772144415486238, 0.0026003244047582644, 0.0017925313204398686, 0.0012356798747410929, 0.000851814824895487, 0.0005871977934930428, 0.00040478427776295553, 0.0002790376825999072, 0.00019235437883366384, 0.00013259932032024755, 9.140722377105822e-05, 6.3011488575907e-05, 4.343691372244476e-05], [1, 0.6982167352537723, 0.4875066093884363, 0.34038527322183, 0.23766269419740943, 0.16594007043413087, 0.11586213422629987, 0.08089688109902145, 0.05648355621317136, 0.03943776421468344, 0.027536106975684323, 0.019226170714160933, 0.013424034147473134, 0.009372885296383849, 0.006544305371549217, 0.00456934353102682, 0.00319039212248649, 0.0022275851719418703, 0.0015553372462529657, 0.001085962494297338, 0.0007582371873763307, 0.0005294138935179045, 0.0003696456403300595, 0.0002580927721920443, 0.0001802046927925248, 0.00012582193227900565, 8.785097877916855e-05, 6.13390235920395e-05, 4.282793279608793e-05], [1, 0.7066326530612245, 0.49932970637234486, 0.3528426751661723, 0.24933015566589217, 0.1761848293863575, 0.12449795341842099, 0.08797431912475157, 0.062165526524378026, 0.04392819093686917, 0.031041094105899898, 0.021934650681975185, 0.015499740405375323, 0.010952622684410624, 0.00773948082546363, 0.005468969869013841, 0.0038645526880531477, 0.0027308191188538822, 0.0019296859589860342, 0.0013635791087732948, 0.0009635495232913334, 0.000680875555999233, 0.0004811289005402744, 0.0003399813914532041, 0.00024024195263402432, 0.00016976280836638962, 0.00011995994366706614, 8.476761325453398e-05, 5.989956344771917e-05, 4.232698743627095e-05]]
    # plotting
def plot_improved_elements(ax, step, euler):
    ie = ImprovedEuler()
    # initialise ODE
    x = [0,]
    y = [1,]
    h0 = 0.1
    
    xlim = np.array([-0.05,1.15])
    ylim = [-0.9,10]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)    
    
    i = int(np.floor(step))
    x = ie.x[:i+1]
    y = ie.y[:i+1]
    dydx0 = ie.dydx0s[i]
    dydx1 = ie.dydx1s[i]
    dydxm = 0.5*(ie.dydx0s[i-1] + ie.dydx1s[i-1])
    yp = ie.yps[i]
    
    if not euler:
        j = abs(step-int(step))
        if 0.2 < j < 0.4:
            # plot derivative
            ys = dydx0*(xlim - x[-1])+y[-1]
            ax.plot(xlim, ys, 'r--')        
            ax.text(0.95*xlim[-1], np.min([1.05*ys[-1],9.]), 'predictor derivative: $f^{'+'{:d}'.format(int(step))+'}=(x^{'+'{:d}'.format(int(step))+'},y^{'+'{:d}'.format(int(step))+'})$', ha = 'right', va = 'bottom', color = 'r', size = TEXTSIZE)
            
            ax.text(0.02, 0.95, 'PREDICTOR STEP', color = 'k', transform=ax.transAxes, ha = 'left', va = 'top', size = TEXTSIZE)
        elif 0.4 < j < 0.6:
            ax.plot([x[-1],x[-1]+h0],[y[-1],yp],'k--o', mfc = 'w')
            dy = 0.4
            dx = 0.04
            ax.arrow(x[-1], y[-1]-dy, h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.01, color= 'r', linewidth = 0.5)
            ax.arrow(x[-1]+h0, y[-1]-dy, -h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.01, color= 'r', linewidth = 0.5)
            ax.text(0.5*(2*x[-1]+h0), y[-1]-2*dy, '$x^{'+'{:d}'.format(int(step+1))+'}=x^{'+'{:d}'.format(int(step))+'}+h$', ha = 'center', va = 'top', color = 'r', size = TEXTSIZE)
            
            ax.arrow(x[-1]+h0+dx, yp, 0, y[-1]-yp, length_includes_head = True, head_width = 0.01, head_length = 0.2, color= 'r', linewidth = 0.5)
            ax.arrow(x[-1]+h0+dx, y[-1], 0, yp-y[-1], length_includes_head = True, head_width = 0.01, head_length = 0.2, color= 'r', linewidth = 0.5)        
            ax.text(x[-1]+h0+2*dx, 0.5*(y[-1]+yp), '$y_p^{'+'{:d}'.format(int(step+1))+'}=y^{'+'{:d}'.format(int(step))+'}+hf^{'+'{:d}'.format(int(step))+'}$', ha = 'left', va = 'center', color = 'r', size = TEXTSIZE)
            
            ax.text(0.02, 0.95, 'PREDICTOR STEP', color = 'k', transform=ax.transAxes, ha = 'left', va = 'top', size = TEXTSIZE)
            
        elif 0.6 < j < 0.8:
            ax.plot([x[-1],x[-1]+h0],[y[-1],yp],'k--o', mfc = 'w')
            # plot derivative
            ys = dydx0*(xlim - x[-1])+y[-1]
            ax.plot(xlim, ys, 'r--')    
            ys = dydx1*(xlim - x[-1]-h0)+yp
            ax.plot(xlim, ys, 'b--')    
            
            ax.text(0.95*xlim[-1], np.min([1.05*ys[-1],9.]), 'corrector derivative: $f_p^{'+'{:d}'.format(int(step+1))+'}=(x^{'+'{:d}'.format(int(step+1))+'},y_p^{'+'{:d}'.format(int(step+1))+'})$', ha = 'right', va = 'bottom', color = 'b', size = TEXTSIZE)
            
            ax.text(0.02, 0.95, 'CORRECTOR STEP', color = 'k', transform=ax.transAxes, ha = 'left', va = 'top', size = TEXTSIZE)
        else:    
            dy = 0.4
            dx = 0.04

            ys = dydxm*(xlim - x[-1])+y[-1]
            ax.plot(xlim, ys, 'g--')    
            
            
            ax.text(0.95*xlim[-1], np.min([1.05*ys[-1],9.]), 'average derivative: '+r'$\frac{h}{2}(f^{'+'{:d}'.format(int(step-1))+'}+f_p^{'+'{:d}'.format(int(step))+'})$', ha = 'right', va = 'bottom', color = 'g', size = TEXTSIZE)
            
            ax.arrow(x[-2], y[-2]-dy, h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.01, color= 'g', linewidth = 0.5)
            ax.arrow(x[-1], y[-2]-dy, -h0, 0, length_includes_head = True, head_width = 0.2, head_length = 0.01, color= 'g', linewidth = 0.5)
            ax.text(0.5*(x[-1]+x[-2]), y[-2]-2*dy, '$x^{'+'{:d}'.format(int(step))+'}=x^{'+'{:d}'.format(int(step-1))+'}+h$', ha = 'center', va = 'top', color = 'g', size = TEXTSIZE)
            
            ax.arrow(x[-1]+dx, y[-2], 0, y[-1]-y[-2], length_includes_head = True, head_width = 0.01, head_length = 0.2, color= 'g', linewidth = 0.5)
            ax.arrow(x[-1]+dx, y[-1], 0, -y[-1]+y[-2], length_includes_head = True, head_width = 0.01, head_length = 0.2, color= 'g', linewidth = 0.5)        
            ax.text(x[-1]+2*dx, 0.5*(y[-1]+y[-2]), '$y^{'+'{:d}'.format(int(step))+'}=y^{'+'{:d}'.format(int(step-1))+r'}+\frac{h}{2}(f^{'+'{:d}'.format(int(step-1))+'}+f_p^{'+'{:d}'.format(int(step))+'})$', ha = 'left', va = 'center', color = 'g', size = TEXTSIZE)
                    
            ax.text(0.02, 0.95, 'CORRECTOR STEP', color = 'k', transform=ax.transAxes, ha = 'left', va = 'top', size = TEXTSIZE)
        
    ax.plot(x,y,'ko-', mfc = 'k', label='Improved Euler', zorder = 2)
    
    ax.plot(x[-1],y[-1],'ko', mfc = 'w', zorder = 3)
    
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    
    
    if euler:
        # plot euler for comparison
        x0 = [0,]
        y0 = [1,]
        for i in range(int(np.floor(step))):
            y0.append(y0[-1]+h0*dydx(x0[-1], y0[-1]))
            x0.append(x0[-1]+h0)        
            
        ax.plot(x0,y0,'ko-', color = [0.7,0.7,0.7], mec = [0.7,0.7,0.7], zorder = 1, label = 'Euler')
        
        ax.legend(loc=2,prop={'size':TEXTSIZE})
        
        
# BACKWARD EULER EXERCISE
# -----------------------
# an exercise to give an idea about how backward Euler works
    # analytic function (true solution)
def logistic(x, *p): return 1/(1+np.exp(-p[0]*x))
    # ODE to solve
def dlogistic(x, y, *p): return p[0]*(1-y)/(1+np.exp(-p[0]*x))
    # root equation for minimization
def root_equation(yk1, yk, h, xk, f, *p):
    return yk - yk1 + h*f(xk+h, yk1, *p) 
    # implement backward Euler method
def beuler(xs, x0, y0, p):
    ys = [y0,]
    for x in xs:
        ynew = root(root_equation, ys[-1], args = (ys[-1], p[2], x-p[2], dlogistic, *p))
        ys.append(ynew.x)
    return ys[1:]
    # check if guess accurate to within tolerance
def check(yest, ytrue):
    return bool(abs(yest - ytrue)/abs(ytrue) < 0.5e-2)
    # linked widgets for user input
def linked_widgets(p):    
    items = [FloatText(value = 0.01, description='guess $y_1$'),
             FloatText(value = 0.9, description='guess $y_2$'),
             FloatText(value = 0.9, description='guess $y_3$'),
             FloatText(value = 0.9, description='guess $y_4$'),
             Checkbox(value = False, description='show $y$'),]
        
    items[1].disabled=True
    items[2].disabled=True
    items[3].disabled=True
    items[4].disabled=True
    
    x0 = p[1]
    y0 = logistic(x0,*p)
    h = p[2]
    xs = h*np.arange(5)[1:]+x0    
    y1, y2, y3, y4 = beuler(xs, x0, y0, p)
    
    def box1_change(change):
        if check(change.new, y1):
            items[1].disabled = False
        else:
            items[1].disabled = True
    items[0].observe(box1_change, names = 'value')
    def box1_disabled(change):
        if change.new: 
            items[2].disabled = True
            items[3].disabled = True
        else:
            items[2].disabled = (not check(items[1].value, y2))
            if not items[2].disabled:
                items[3].disabled = (not check(items[2].value, y3))
    items[1].observe(box1_disabled, names = 'disabled')
    
    def box2_change(change):
        if check(change.new, y2): 
            items[2].disabled = False
        else:
            items[2].disabled = True
    items[1].observe(box2_change, names = 'value')
    def box2_disabled(change):
        if change.new: 
            items[3].disabled = True
        else:
            items[3].disabled = not check(items[2].value, y3)            
    items[2].observe(box2_disabled, names = 'disabled')
    
    def box3_change(change):
        if check(change.new, y3): 
            items[3].disabled = False
        else:
            items[3].disabled = True            
    items[2].observe(box3_change, names = 'value')
    
    def box4_change(change):
        if check(change.new, y4): 
            items[4].disabled = False
        else:
            items[4].disabled = True            
    items[3].observe(box4_change, names = 'value')
        
    items[4].description = 'plot $y$'
    
    return items
    # plotting
def plot_beuler_elements(ax, guess_y1, guess_y2, guess_y3, guess_y4, plot_true, p):
    # initialise ODE
    x0 = p[1]
    y0 = logistic(x0,*p)
    h = p[2]
    xs = h*np.arange(5)+x0    
    ys = np.array([y0,]+ list(beuler(xs[1:], x0, y0, p)))
    y1, y2, y3, y4 = ys[1:]
    
    # set up limits and plot vertical lines
    xm = 0.5*(x0+xs[-1])
    xr = (xs[-1] - x0)
    ym = 0.5*(np.min(ys) + np.max(ys))
    yr = (np.max(ys) - np.min(ys))
    xlim = [xm-0.6*xr, xm+0.6*xr]
    ylim = [ym-0.6*yr, ym+0.6*yr]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.plot([x0,x0], ylim, 'k:')
    for x in xs:
        ax.plot([x,x], ylim, 'k:')
    
    # how many correct guesses so far?
    gys = [guess_y1, guess_y2, guess_y3, guess_y4]
    i1 = 0
    for x, gy, y in zip(xs[1:], gys, ys[1:]):
        if check(gy, y):
            i1 += 1
        else:
            break
    
    # optional plotting of true function
    xv = np.linspace(xlim[0], xlim[1], 1001)
    if True:        
        yv = logistic(xv, *p)
        #ax.plot(xv,yv, 'k-', label = 'analytical solution')
    
    # plot correct steps so far
    ax.plot(xs[:i1+1], ys[:i1+1], 'k-o', label='numerical solution')
    
    # plot previous guess
    if i1 > 0 and not plot_true:
        m = (ys[i1]-ys[i1-1])/(xs[i1]-xs[i1-1])
        ax.plot(xv, m*(xv-xs[i1])+gys[i1-1],'k-',alpha = 0.5, label = 'prior gradient')
        
        x2 = np.array([x-2*h, x-h])
        if i1 == 4:
            x2 += h
        ax.plot(x2, m*(x2-xs[i1])+gys[i1-1],'k-',lw = 2)
        
    # plot next guess
    if i1 < 4:
        m = dlogistic(x, gy, *p)
        ax.plot(xv, m*(xv-x)+gy,'r-',alpha = 0.5, label='predicted gradient')        
        x2 = np.array([x-h, x])
        ax.plot(x2, m*(x2-x)+gy,'r-o',lw = 2, mec = 'r')
    
    # plot Euler steps
    if plot_true:    
        yv = logistic(xv, *p)
        ax.plot(xv,yv, 'b-', label = 'analytical solution')
        
    ax.legend(loc = 4, prop = {'size':TEXTSIZE})
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
    for t in ax.get_xticklabels()+ax.get_yticklabels():    t.set_fontsize(TEXTSIZE)

    
# BACKWARD EULER METHOD
# ---------------------
# demonstration of backward Euler method
    # plotting
def plot_beuler_elements2(ax, step, euler):    
    # initialise ODE
    x = [0,]
    y = [1,]
    h0 = 0.08
    
    xlim = np.array([-0.05,0.75])
    ylim = [0.1,6]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)    
    
    for i in range(int(np.ceil(step))):
        ynew = root(root_equation, y[-1], args = (y[-1], h0, x[-1], dydx))
        y.append(ynew.x)
        x.append(x[-1]+h0)
    
    if not euler:
        j = abs(step-int(step))
        
        xi,yi = [x[-1], y[-1]]
        x = x[:-1]
        y = y[:-1]
        
        dy1 = 0.2*(yi-y[-1])
        dy2 = 1.5*(yi-y[-1])
        dy3 = 0.7*(yi-y[-1])
        dy4 = (yi-y[-1])
        y1 = y[-1]+dy1
        y2 = y[-1]+dy2
        y3 = y[-1]+dy3
        y4 = y[-1]+dy4
        dydx1 = dydx(xi, y1)
        dydx2 = dydx(xi, y2)
        dydx3 = dydx(xi, y3)
        dydx4 = dydx(xi, y4)
        
        dx = 0.05
        dy = 0.7
        
        if step < 3.25:
            ha = 'left'
        elif step < 4.25:
            ha = 'center'
        else: 
            ha = 'right'
        
        if 0.2 < j < 0.4:
            ys = dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--')    
            ax.plot(xi,y1,'bo', mfc = 'w')
            
            ys = dydx1*(x[-1] - xi)+y1
            ax.arrow(x[-1],ys[0],0,y[-1][0]-ys[0],length_includes_head = True, head_length = 0.12, head_width=0.01, color = 'b')
            ax.arrow(x[-1],y[-1][0],0,-y[-1][0]+ys[0],length_includes_head = True, head_length = 0.12, head_width=0.01, color = 'b')
            
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $y^{'+'{:d}'.format(int(step)+1)+'}$: undershoot $y^{'+'{:d}'.format(int(step))+'}$', color = 'b', size = TEXTSIZE, ha = ha, va = 'center')
            
        elif 0.4 < j < 0.6:
            ys = dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--', alpha = 0.5)
            ys = dydx2*(xlim - xi)+y2
            ax.plot(xlim, ys, 'b--')
            ax.plot(xi,y2,'bo', mfc = 'w')
            
            ys = dydx2*(x[-1] - xi)+y2
            ax.arrow(x[-1],ys[0],0,y[-1][0]-ys[0],length_includes_head = True, head_length = 0.12, head_width=0.01, color = 'b')
            ax.arrow(x[-1],y[-1][0],0,-y[-1][0]+ys[0],length_includes_head = True, head_length = 0.12, head_width=0.01, color = 'b')
            
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $y^{'+'{:d}'.format(int(step)+1)+'}$: overshoot $y^{'+'{:d}'.format(int(step))+'}$', color = 'b', size = TEXTSIZE, ha = ha, va = 'center')
            
        elif 0.6 < j < 0.8:
            ys = dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--', alpha = 0.5)
            ys = dydx2*(xlim - xi)+y2
            ax.plot(xlim, ys, 'b--', alpha = 0.5)
            ys = dydx3*(xlim - xi)+y3
            ax.plot(xlim, ys, 'b--')
            ax.plot(xi,y3,'bo', mfc = 'w')
            
            ys = dydx3*(x[-1] - xi)+y3
            ax.arrow(x[-1],ys[0],0,y[-1][0]-ys[0],length_includes_head = True, head_length = 0.12, head_width=0.01, color = 'b')
            ax.arrow(x[-1],y[-1][0],0,-y[-1][0]+ys[0],length_includes_head = True, head_length = 0.12, head_width=0.01, color = 'b')
            
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $y^{'+'{:d}'.format(int(step)+1)+'}$: undershoot $y^{'+'{:d}'.format(int(step))+'}$', color = 'b', size = TEXTSIZE, ha = ha, va = 'center')
        else:    
            ys = dydx1*(xlim - xi)+y1
            ax.plot(xlim, ys, 'b--', alpha = 0.5)
            ys = dydx2*(xlim - xi)+y2
            ax.plot(xlim, ys, 'b--', alpha = 0.5)
            ys = dydx3*(xlim - xi)+y3
            ax.plot(xlim, ys, 'b--', alpha = 0.5)
            ys = dydx4*(xlim - xi)+y4
            ax.plot(xlim, ys, 'k--')
            
            ax.plot(xi,yi,'ko', mfc = 'w', zorder = 3)
    
            ax.text(x[-1]-dx, y[-1]+dy, 'solving for $y^{'+'{:d}'.format(int(step))+'}$: within tolerance', color = 'k', ha = ha, va = 'center', size = TEXTSIZE)
            
    ax.plot(x,y,'ko-', mfc = 'k', label='Backward Euler', zorder = 2)
    
    
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    
    if euler:
        # plot euler for comparison
        x0 = [0,]
        y0 = [1,]
        while len(x0) < len(x):
            y0.append(y0[-1]+h0*dydx(x0[-1], y0[-1]))
            x0.append(x0[-1]+h0)        
            
        ax.plot(x0,y0,'ko-', color = [0.7,0.7,0.7], mec = [0.7,0.7,0.7], zorder = 1, label = 'Euler')
        
        ax.legend(loc=2,prop={'size':TEXTSIZE})
    
    
# STABILITY OF ALL METHODS
# ------------------------
# demonstrate how the stability of all methods are sensitive to step size
    # ODE to solve
def dydx2(x,y): return -10*y
    # plotting
def plot_stability_elements(ax, step, method):
    # initialise ODE
    x0,x1 = [0,1]
    y0 = 1.
    
    h = x1/step
    
    if method == 'Euler':
        x = [x0,]
        y = [y0,]
        while x[-1] < x1:        
            y.append(y[-1]+h*dydx2(x[-1],y[-1]))
            x.append(x[-1]+h)
            
        ax.plot(x,y,'b--x', label='Euler')

    elif method == 'RK45':
        x = np.linspace(x0,x1, step+1)
        y = DoPri45integrate(dydx2, x, y0) 
            
        ax.plot(x,y,'r--x', label='RK45')
    
    elif method == 'Improved Euler':
        x = [x0,]
        y = [y0,]
        while x[-1] < x1:        
            y.append(y[-1]+h/2.*(dydx2(x[-1],y[-1])+dydx2(x[-1]+h,y[-1]+h*dydx2(x[-1],y[-1]))))
            x.append(x[-1]+h)
            
        ax.plot(x,y,'g--x', label='Improved Euler')
    
    xv = np.linspace(x0,y0,101)
    yv = np.exp(-10*xv)
    ax.plot(xv,yv,'c-', lw=2, label='exact')    
    ax.legend(loc=1, prop={'size':TEXTSIZE})
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim([x0,x1])
    ax.set_xlabel('x', size = TEXTSIZE)
    ax.set_ylabel('y', size = TEXTSIZE)
    
    ax.text(0.5, 0.95, 'a=-10, h={:4.3f}'.format(h), transform=ax.transAxes, ha = 'center', va = 'top', size = TEXTSIZE)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    
    
# ACCURACY OF ALL METHODS
# -----------------------
# demonstrate how the accuracy of all methods are sensitive to step size
    # plotting
def plot_accuracy_elements(ax, ax2, step_ind):

    steps = np.logspace(np.log10(3), np.log10(100),16)
    steps = [int(np.round(step)) for step in steps]
    step = steps[step_ind]

    ie = ImprovedEuler()
    # initialise ODE
    x0,x1 = [0,0.2]
    y0 = 1.
    
    xv = np.linspace(x0,x1,101)
    yv = np.exp(-10*xv)
    
    h = x1/step
    
    x = np.arange(step+1)*h
    y = 0*x
    y[0] = y0
    for i in range(step):
        y[i+1] = y[i]+h*dydx2(x[i],y[i])
        
    ax.plot(x,y,'b--x', label='Euler')
    
    ym = 0.5*(y[-1]+yv[-1])
    
    y = DoPri45integrate(dydx2, x, y0)  
    ax.plot(x,y,'r--x', label='RK45')
    
    for i in range(step):
        y[i+1]=y[i]+h/2.*(dydx2(x[i],y[i])+dydx2(x[i]+h,y[i]+h*dydx2(x[i],y[i])))
    ax.plot(x,y,'g--x', label='Improved Euler')
    
    ax.plot(xv,yv,'c-', lw=2, label='exact')    
    ax.legend(loc=1, prop={'size':TEXTSIZE})
    
    ax.set_xlim([x0,1.05*x1])
    ylim = [0,1.05]
    ax.set_ylim(ylim)
    
    xf,yf = [0.7,0.3]
    
    xa = 0.85*x1
    ya = 0.5*(ylim[1]-ylim[0])+ylim[0]
    ax.arrow(xa,ya,x1-xa,ym-ya,length_includes_head=True,head_length=0.02,head_width=0.004, color='k', linewidth=0.5)
    ax.text(xa, ya, 'error ', size = TEXTSIZE, ha='right', va = 'center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
    
    errE = []
    errB = []
    errI = []
    hs = []
    
    for step in steps:        
        h = x1/step
        hs.append(h)
        x = np.linspace(x0,x1,step+1)
        y = 0.*x
        y[0] = y0
        for i in range(step):
            y[i+1] = y[i] + h*dydx2(x[i],y[i])
        errE.append(abs(y[-1] - np.exp(-10*x1)))
        
        # backward Euler
        y = DoPri45integrate(dydx2, x, y0)  
        errB.append(abs(y[-1] - np.exp(-10*x1)))
        
        # improved Euler
        for i in range(step):
            y[i+1] = y[i]+0.5*h*(dydx2(x[i],y[i])+dydx2(x[i]+h,y[i] + h*dydx2(x[i],y[i])))
        errI.append(abs(y[-1] - np.exp(-10*x1)))
        
    ax2.plot(hs, errE, 'b.', mfc = 'b', label = 'Euler')
    ax2.plot(hs, errB, 'r.', mfc = 'r', label = 'RK45')
    ax2.plot(hs, errI, 'g.', mfc = 'g', label = 'Improved Euler')
    ax2.plot(hs[step_ind], errE[step_ind], 'bo', mfc = 'b')
    ax2.plot(hs[step_ind], errB[step_ind], 'ro', mfc = 'r')
    ax2.plot(hs[step_ind], errI[step_ind], 'go', mfc = 'g')
    ax2.legend(loc=2, prop={'size':TEXTSIZE})
    
    ax2.set_xlabel('step size', size = TEXTSIZE)
    ax2.set_ylabel('error', size = TEXTSIZE)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    for t in ax2.get_xticklabels()+ax2.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    # a copy-pasted implementation from Stack Overflow
def DoPri45Step(f,t,x,h):

    k1 = f(t,x) ;
    k2 = f(t + 1./5*h, x + h*(1./5*k1) ) ;
    k3 = f(t + 3./10*h, x + h*(3./40*k1 + 9./40*k2) ) ;
    k4 = f(t + 4./5*h, x + h*(44./45*k1 - 56./15*k2 + 32./9*k3) ) ;
    k5 = f(t + 8./9*h, x + h*(19372./6561*k1 - 25360./2187*k2 + 64448./6561*k3 - 212./729*k4) ) ;
    k6 = f(t + h, x + h*(9017./3168*k1 - 355./33*k2 + 46732./5247*k3 + 49./176*k4 - 5103./18656*k5) )

    v5 = 35./384*k1 + 500./1113*k3 + 125./192*k4 - 2187./6784*k5 + 11./84*k6;
    k7 = f(t + h, x + h*v5);
    v4 = 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 - 92097./339200*k5 + 187./2100*k6 + 1./40*k7; 

    return v4,v5
def DoPri45integrate(f, x, y0):
    N=len(x)
    y = np.asarray(N*[y0])
    for k in range(N-1):
        v4, v5 = DoPri45Step(f,x[k],y[k],x[k+1]-x[k])
        y[k+1] = y[k] + (x[k+1]-x[k])*v5
    return y    
    
# STIFFNESS OF ODES
# -----------------
# demonstrate how solution stiffness manifests as step size increases in Euler method    
    # ODE to solve (exponential with variable coefficient)
def df(x,y,*p):
    if x<p[0]:
        return p[2]*y
    elif x<(p[0]+p[1]):
        return p[3]*y
    else:
        return p[2]*y
    # plotting
def plot_stiffness_elements(ax, ax2, h):
    # initialise ODE
    x0 = 20
    L = 20
    x = np.linspace(0,x0+2*L, 1000)

    a = -0.1
    a2 = -1

    i1 = np.where((x<x0))
    i2 = np.where((x>=x0)&(x<(x0+L)))
    i3 = np.where((x>=(x0+L)))
    y = 0.*x
    y[i1] = np.exp(a*x[i1])
    y[i2] = np.exp(a2*x[i2]+(a-a2)*x0)
    y[i3] = np.exp(a*x[i3]+(a2-a)*L)

    ax.plot(x,y,'b-',label='analytic solution',zorder=3)
    p = np.array([x0,L,a,a2])
    x1 = x0+2*L
    x0,y0=[0,1]
    xs = [x0,]
    ys = [y0,]
    while xs[-1] < x1:        
        ys.append(ys[-1]+h*df(xs[-1],ys[-1], *p))
        xs.append(xs[-1]+h)
        
    x0,x1 = [0,1]
    y0 = 1
    
    ax.plot(xs,ys,'ro-',label='Euler method')
    
    ax.legend(loc=3, prop={'size':TEXTSIZE})

    #ax.set_yscale('symlog',linthreshy=1.e-30)
    #ax.set_ylim([-1.e5,1.e5])
    ax.set_ylim([-1,1])
    ax2.plot(x,[-df(xi,1,*p) for xi in x],'b-', label='-a(x)')
    ax2.plot(x,[-2*df(xi,1,*p) for xi in x],'b--', label='-2a(x)')
    
    xlim = ax2.get_xlim()
    ax2.set_xlim(xlim)
    ax2.plot(xlim, [2./h, 2./h], 'r--',label='2/h')    
    
    #ax2.set_yscale('symlog',linthreshy=2.)
    ax2.set_ylim([0,3])
    ax2.legend(prop={'size':TEXTSIZE})
    
    ax2.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    for t in ax2.get_xticklabels()+ax2.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    
    
# CONVERGENCE OF ODES
# -------------------
# demonstrate how solution converges for decreasing step size
    # plotting
def dydx3(x,y): return -10*y
def plot_convergence_elements(ax, ax2, step_ind, zoom):

    steps = np.logspace(np.log10(3), np.log10(150),16)
    steps = [int(np.round(step)) for step in steps]
        
    # initialise ODE
    x0,x1 = [0,0.2]
    y0 = 1
    
    hs = []
    ys = []
    for step in steps[:step_ind+1]:
    
        h = x1/step
        hs.append(h*1.)
    
        x = np.arange(step+1)*h
        y = 0*x
        y[0] = y0
        for i in range(step):
            y[i+1] = y[i]+h*dydx3(x[i],y[i])
            
        ys.append(y[-1]*1)
        
        ax.plot(x,y,'b-', lw = 0.5, alpha=0.5)
    
    ax.plot(x,y,'b-')
    
    ax.set_xlim([x0,1.05*x1])
    ylim = [0,1.05]
    ax.set_ylim(ylim)
    
    bbox = [0.18,0.21,0.03,0.145]
    if zoom:
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])
    else:
        ax.plot([bbox[0],bbox[1],bbox[1],bbox[0],bbox[0]],[bbox[2],bbox[2],bbox[3],bbox[3],bbox[2]], 'k-')
        ax.text(0.5*(bbox[0]+bbox[1]), bbox[3], 'zoom',ha = 'center',va='bottom',size=TEXTSIZE)
    
    #ax.set_xticks([])
    #ax.set_yticks([])
    
    ax.set_xlabel('$x$', size = TEXTSIZE)
    ax.set_ylabel('$y(x)$', size = TEXTSIZE)
        
    ax2.plot(1./np.array(hs), ys, 'bs', mfc = 'b')
    
    ax2.set_xlabel('$1/h$', size = TEXTSIZE)
    ax2.set_ylabel('y(0.2)', size = TEXTSIZE)
    ax2.set_xscale('log')
    ax2.set_xlim([9,1001])
    ax2.set_ylim([0.03,0.145])
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
    for t in ax2.get_xticklabels()+ax2.get_yticklabels(): t.set_fontsize(TEXTSIZE)
   