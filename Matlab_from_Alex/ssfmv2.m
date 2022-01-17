function [t,Ats,totalWs,Aws]=ssfmv2(t,pump,center_wl,alpha,ldas,n,L,zSpace,chi2,gratingProfile,gamma,fR,numSteps)

    c = 299792458;
 
    w = 2*pi*c./(ldas*1e-9);
    beta = n.*w/c;

    N = length(t);
    tMax = max(t);
    tMin = min(t);
    tW = tMax - tMin;
    dT = abs(t(2)-t(1));
    
    
    %% Pump parameters
    ref_freq = 2*pi*c/center_wl/1e12;    
    ws = (2*pi/tW)*(-N/2:N/2-1); % relative angular frequency
    totalWs = ws + ref_freq;
    At = pump;
    
    %% Setting up the simulation parameters
    [~,center_index] = min(abs(ws));
    betasim = spline(w/1e12, beta, totalWs); 
    slope = gradient(betasim)./gradient(totalWs);
    betaref = betasim(center_index);
    beta1 = slope(center_index);
    beta2sim = betasim - betaref - beta1*ws;
    beta2sim = beta2sim + 1j*alpha;
    
    z = zSpace(1);
    dz = 2e-6;
    A  = At;
    R = ramanResponse();
    
   

    % Pre-shift everything for faster computation
    ws = fftshift(ws);
    beta2sim = fftshift(beta2sim);
    A = fftshift(A);
    Aw = fft(ifftshift(A));
    R = fftshift(R);
    Ats = zeros(numSteps,length(A));
    Aws = zeros(numSteps,length(A));
    

    figure;
    
   % set(gcf, 'Position', get(0,'Screensize')); 
    [~,indices] = find(totalWs > 0);
    semilogy(c*1e-6./(totalWs(1,indices)/(2*pi)),abs(Aw(1,indices)).^2./max(abs(Aw(1,indices)).^2));
    %semilogy(t, abs(fftshift(At)).^2);
    %xlim([-20,20])
    xlim([1,15])
    ylim([1e-6,1])
    set(gca,'YScale','log');
    set(gca,'NextPlot', 'replaceChildren');
    
    function [A,dz] = integrate(A,dz,zEnd)
        while z < zEnd
            
            Ac = propagate(A,z,dz);
            Af1 = propagate(A,z,dz/2);
            Af = propagate(Af1,z,dz/2);

            % u4 = (4/3)*Af - (1/3)*Ac; local error
            le = 1e-3;
            dle = norm(Af-Ac);
            if norm(Af) ~= 0
                dle = norm(Af-Ac)/norm(Af); % relative local error 
            end

            old_dz = dz;

            if  dle > 2*le 
                dz = dz/2;
                continue;
            elseif dle < 2*le && dle > le
                dz = dz/2^(1/3.); 
            elseif dle < 0.5*le
                dz = dz*2^(1/3);
            end

            A = (4/3)*Af - (1/3)*Ac;
            z = z+old_dz;
            Aw = fftshift(ifft(A));
            semilogy(c*1e-6./(totalWs(1,indices)/(2*pi)),abs(Aw(1,indices)).^2./max(abs(Aw(1,indices)).^2),'LineWidth',2)
            %semilogy(t, abs(fftshift(A)).^2./max(abs(fftshift(A)).^2));
            title(sprintf('Z Position: %4.3f mm',z/1e-3))


            drawnow
            clc
            z, sum(dT*abs(A).^2)


        end
    end

    zs = linspace(zSpace(1),zSpace(end),numSteps);
    
    for ii=1:numSteps
        [A,dz] = integrate(A,dz,zs(ii));
        Ats(ii,:) = A;
        Aws(ii,:) = fftshift(ifft(A));
    end

    %endAw = fftshift(fft(fftshift(A)));

    
    %% Workhorse functions

    
   function factor=gBeamApprox(z)
       z_0 = pi*(15e-6).^2/(center_wl);
       factor = 1/sqrt(1+((z-L/2)/z_0).^2);
   end
    
    function dX=deriv(X)
        Xw = ifft(X);
        dX = fft(-1j*ws.*Xw);
    end

    function dispX=dispX(X,dz)
        Xw = ifft(X);
        dispX = fft(exp(1j*0.5*beta2sim.*dz).*Xw);
    end

    function raman=ramanResponse()
        % PPLN Raman Response
        function RT=R(tau1,tau2)
                RT = ((tau1^2+tau2^2)./(tau1*tau2^2)).*exp(-T/tau2).*sin(T/tau1);
                RT(1:length(RT)/2)=0;
                RT = RT./trapz(T,RT);
        end
        
        if fR > 0
            tau1s = [0.021,0.0193,0.0159,0.083];
            tau2s = [0.544,1.021,1.361,0.544];
            T = t;
            weights = [0.635,0.105,0.020,0.240];
           

            RT = R(tau1s(1),tau2s(1))*weights(1) + ...
                R(tau1s(2),tau2s(2))*weights(2) + ...
                R(tau1s(3),tau2s(3))*weights(3) + ...
                R(tau1s(4),tau2s(4))*weights(4);


            raman = ((1.-fR)+tW.*ifftshift(ifft(fftshift(fR*RT))));
        else
            raman = 1;
        end
    end

    function raman=silicaRamanResponse()
        function RT=Rt(tau1,tau2)
                RT = ((tau1^2+tau2^2)./(tau1*tau2^2)).*exp(-T/tau2).*sin(T/tau1);
                RT(1:length(RT)/2)=0;
                RT = RT./trapz(T,RT);
        end
        
        if fR > 0
            tau1s = 0.0122;
            tau2s = 0.0320;
            T = t;
            weights = 1;
           

            RT = Rt(tau1s,tau2s)*weights;


            raman = ((1.-fR)+tW.*ifftshift(ifft(fftshift(fR*RT))));
            
            
            figure;
            subplot(2,2,1);
            plot(ws/(2*pi), abs(raman - (1-fR)))
            subplot(2,2,2);
            plot(ws/(2*pi), unwrap(angle(raman -(1-fR))))
            subplot(2,2,3);
            plot(t*1000, ifftshift(real(fft(raman - (1-fR)))))
            
            
        else
            raman = 1;
        end
        
    end

    
    function nA = nonlinear(X,z)    
         chi = chi2.*grating(z).*gBeamApprox(z);        

         phase = -(1j*ref_freq*t - 1j*betaref*z + 1j*beta1*ref_freq*z);
         expPhi = exp(phase);
         expPhi = fftshift(expPhi);
        
         dX = deriv(X);
         X2 = abs(X).^2;
         dX2 = deriv(X2);
         % with self-steepening
         nLTerm = 2*X.*expPhi + (2j/ref_freq)*dX.*expPhi + ...
             (4j/ref_freq)*real(conj(X).*dX).*conj(expPhi)./(X+1e-20);

         % without self-steepening
         % nLTerm = X.*expPhi + 2*conj(X).*conj(expPhi);
         
         nA = 1j*chi*nLTerm;
         
         % chi3 terms -- from PyNLO 
         
         X2w = ifft(X2);
         R_X2 = fft(R.*X2w);
         dR_X2 = fft(R.*ifft(dX2));
         
         nA = nA  + 1j*gamma*R_X2 - (gamma/ref_freq)*(dR_X2+(dX.*R_X2./(X+1e-20))); 
         
        
         
    end

    function Xn = propagate(X,z,dz) 
        % RK4 + SSFM -- not RK4IP, not in the interaction picture
        Xd = dispX(X,dz);
        k1 = dz*nonlinear(Xd,z).*Xd;
        k2 = dz*nonlinear(Xd+0.5*k1,z).*(Xd+0.5*k1);
        k3 = dz*nonlinear(Xd+0.5*k2,z).*(Xd+0.5*k2);
        k4 = dz*nonlinear(Xd+k3,z).*(Xd+k3);
        X_RK = Xd + k1./6 + k2./3 + k3./3 + k4./6;
        Xn = dispX(X_RK,dz);
    end

    function g=grating(z)
        zD = zSpace;
        yD = gratingProfile;
        
        gP = interp1(zD,yD,z);
        g = sign(cos(2*pi/gP*z));
    end

end