import numpy as np
import matplotlib.pyplot as plt


def time_synchronizer(signal, OF, typeof='gardner', debug_level=0, axes=None):
    """
    Time Synchronizer using Gardner or Early-Late detection
    
    Parameters:
    - signal: input signal
    - OF: oversampling factor
    - typeof: 'gardner' or 'early-late' detector type
    - debug_level: if 1, plots debugging information
    - axes: matplotlib axes for plotting (optional)
    Returns:
    - time_sync: time synchronized and downsampled signal
    - tao: final timing error estimate
    """
    
    if typeof not in ['gardner', 'early-late']:
        raise ValueError("typeof must be 'gardner' or 'early-late'")
    
    Ki = 0.00002
    Kp = 0.04
    N = len(signal)
    time_sync = np.zeros(N // OF, dtype=complex)
    integrate = 0
    sigma = 1
    T = OF
    tao = 0
    indx = 1
    
    integrate_log = np.zeros((N // OF)-2*OF)
    error_log = np.zeros((N // OF)-2*OF)
    loop_log = np.zeros((N // OF)-2*OF)
    est_delay_log = np.zeros((N // OF)-2*OF)
    
    if typeof == 'gardner':
        T = 2 * OF
        indx = 2
    
    output_idx = 0
    
    while T < N - OF:
        if T + tao - sigma < 1:
            T = T + OF
            continue
        
        if T + tao + sigma > N:
            break
        
        if typeof == 'gardner':  # Gardner Detector
            idx_now = T + tao
            idx_half = T + tao - (OF / 2)
            idx_prev = T + tao - OF
            
            # Linear interpolation
            def interpolate(signal, idx):
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))
                if idx_floor < 0 or idx_ceil >= len(signal):
                    return 0
                frac = idx - idx_floor
                return signal[idx_floor] * (1 - frac) + signal[idx_ceil] * frac
            
            val_now = interpolate(signal, idx_now)
            val_half = interpolate(signal, idx_half)
            val_prev = interpolate(signal, idx_prev)
            
            delta_m = np.real((val_now - val_prev) * np.conj(val_half))
        
        elif typeof == 'early-late':  # Early-Late Gate Detector
            early_idx = T + tao - sigma
            late_idx = T + tao + sigma
            sample_idx = T + tao
            
            def interpolate(signal, idx):
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))
                if idx_floor < 0 or idx_ceil >= len(signal):
                    return 0
                frac = idx - idx_floor
                return signal[idx_floor] * (1 - frac) + signal[idx_ceil] * frac
            
            val_early = interpolate(signal, early_idx)
            val_late = interpolate(signal, late_idx)
            val_now = interpolate(signal, sample_idx)
            
            delta_m = np.abs(val_early) ** 2 - np.abs(val_late) ** 2
        
        integrate = integrate + delta_m * Ki
        loop_filter = integrate + delta_m * Kp
        
        tao = tao - loop_filter
        
        if tao > OF:
            T = T + OF * int(np.floor(tao / OF))
            tao = tao % OF
        
        if tao < -1:
            tao = tao + OF
            T = T - OF
        
        if T > 0 and int(T // OF) < len(time_sync):
            time_sync[int(T // OF)] = val_now
        
        indx = indx + 1
        
        if debug_level and output_idx < len(integrate_log):
            integrate_log[output_idx] = integrate
            error_log[output_idx] = delta_m
            loop_log[output_idx] = loop_filter
            est_delay_log[output_idx] = tao
            output_idx += 1
        
        T = T + OF
    
    if debug_level == 1 and axes is not None:
        axes[0,1].plot(est_delay_log)
        axes[0,1].set_title("Estimated Time Delay (tao)")
        
        axes[1,1].plot(error_log)
        axes[1,1].set_title("Error")
        
        axes[2,1].plot(integrate_log)
        axes[2,1].set_title("Integrate")
        
        plt.tight_layout()
    
    return time_sync, tao
