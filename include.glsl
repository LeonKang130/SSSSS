#define KERNEL_WIDTH 21
#define DMFP 0.6763163208961487
#define WIDTH 0.03
vec4 kernel[KERNEL_WIDTH] = vec4[](
    vec4(0.018091047182679176, 0.018097402527928352, 0.0180776696652174, -10),
    vec4(0.020148461684584618, 0.020128294825553894, 0.02017524652183056, -9),
    vec4(0.022637534886598587, 0.022619809955358505, 0.022591140121221542, -8),
    vec4(0.025625525042414665, 0.025703100487589836, 0.025646934285759926, -7),
    vec4(0.029509371146559715, 0.02945631928741932, 0.02950415015220642, -6),
    vec4(0.03439522534608841, 0.03442388027906418, 0.03438130021095276, -5),
    vec4(0.04110262170433998, 0.04103775694966316, 0.04116227477788925, -4),
    vec4(0.050596028566360474, 0.05073437839746475, 0.050734903663396835, -3),
    vec4(0.06581652164459229, 0.06589063256978989, 0.06575163453817368, -2),
    vec4(0.09447832405567169, 0.0943569466471672, 0.09433413296937943, -1),
    vec4(0.19519869983196259, 0.1951030045747757, 0.19528129696846008, 0),
    vec4(0.09447832405567169, 0.0943569466471672, 0.09433413296937943, 1),
    vec4(0.06581652164459229, 0.06589063256978989, 0.06575163453817368, 2),
    vec4(0.050596028566360474, 0.05073437839746475, 0.050734903663396835, 3),
    vec4(0.04110262170433998, 0.04103775694966316, 0.04116227477788925, 4),
    vec4(0.03439522534608841, 0.03442388027906418, 0.03438130021095276, 5),
    vec4(0.029509371146559715, 0.02945631928741932, 0.02950415015220642, 6),
    vec4(0.025625525042414665, 0.025703100487589836, 0.025646934285759926, 7),
    vec4(0.022637534886598587, 0.022619809955358505, 0.022591140121221542, 8),
    vec4(0.020148461684584618, 0.020128294825553894, 0.02017524652183056, 9),
    vec4(0.018091047182679176, 0.018097402527928352, 0.0180776696652174, 10)
);