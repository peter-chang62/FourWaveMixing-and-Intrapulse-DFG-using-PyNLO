% ppln wrapper function
%clear all;
c = 299792458;
ldas = linspace(300,6000,5000);

n = LNJundt(ldas*0.001);

w = 2*pi*c./(ldas*1e-9);
beta = n.*w/c;

%% Define temporal parameters
Window = 10;
tMin = -Window/2;
tMax = Window/2;
N = 2^14;
t = linspace(tMin,tMax,N);
dT = abs(t(2)-t(1));
tW = tMax - tMin;


%% Pump parameters
center_wl = 1.55e-6; 
center_freq = 2*pi*c/center_wl/1e12;    
ws = (2*pi/tW)*(-N/2:N/2-1); 
totalWs = ws + center_freq;
pulseEnergy = 5e-9; % Jc/
 

% sig = 0.009/1.76; % ps, pulse duration
% At = sech(t/sig);

% expPulse = importdata('O:\OFM\abijith\MATLAB\PPLN Waveguides\Pulse FROG Reconstructions\ExpPulse_Recon_10psRange_2to12pnts.txt');
% expPulseInterp = interp1(expPulse(:,3), exp(-0.5*expPulse(:,3).^2./0.1^2).*abs(expPulse(:,1)+1j*expPulse(:,2))./max(abs(expPulse(:,1)+1j*expPulse(:,2))), t);
% expPulseInterp(isnan(expPulseInterp))=0;
% phase = interp1(expPulse(:,3), angle(expPulse(:,1)+1j*expPulse(:,2)), t);
% phase(isnan(phase))=0;
% At = expPulseInterp.*exp(1j*phase);

pulse = importdata('NDAmplifier.txt');
% pulseInterp = interp1(pulse(:,3), abs(pulse(:,1)+1j*pulse(:,2)).^2./max(abs(pulse(:,1)+1j*pulse(:,2)).^2),t);
% pulseInterp(isnan(pulseInterp))=0;
% phase = interp1(pulse(:,3), angle(pulse(:,1)+1j*pulse(:,2)),t);
% phase(isnan(phase)) = 0;
At = (pulse(:,1)+1j*pulse(:,2))/sqrt(2);

%At = At.*sqrt(pulseEnergy./sum(abs(At).^2*dT*1e-12));
%At = fliplr(At);
%Aw = fftshift(fft(fftshift(At)));
alpha = 1e6*(1+erf(-(totalWs-300.)/(10*sqrt(2)))); % 1000 /m roughly at 5.5 microns according to Schwesyg et al. 
% only implemented the MIR loss, need to add in short-wave (UV/Vis) loss 

deff = 19.6e-12;
chi2 = 2*deff;
e0 = 8.85e-12;
Aeff = pi*15e-6*15e-6; 
n0 = interp1(ldas,n,center_wl*1e9);
chi = (1/4).*(chi2/n0)*(center_freq*1e12/c)*sqrt(2/(e0*c*Aeff));

chi3 = 5200e-24;
gamma = 0;%(3/8)*(chi3/n0)*(center_freq*1e12/c)*(2/(e0*c*Aeff));
%gamma = 0;
fR = 0.0; % this was a free-parameter fit for what made the closest match for the chirped PPLN cascaded-chi2 spectrum vs. experiment, feel free to change it if you think it's contributing nonsense.

% powerFactors = [1.0/sqrt(2), 1, sqrt(2), sqrt(3), sqrt(4), sqrt(5), sqrt(10)];
% pump = At.*powerFactors;

pump = At*sqrt(pulseEnergy/2.3e-9);

zStart = 0e-3;
zEnd = 1e-3; % length of crystal to propagate
zSpace = linspace(zStart, zEnd, 5000);
L = zEnd-zStart;

%gratings = 29.6;
%gratings = [24.06, 24.63, 25.23 25.86, 26.53, 27.22, 27.96, 28.74, 29.56, 30.43, 31.35, 32.33, 33.37, 34.48, 35.67, 36.95];
%gratings = 25.23;
gratings = 25.86;
%chirp = linspace(26.5,29.5,5000); % Design 1
%chirp = linspace(29.5, 26.5, 5000); % Design 2
%chirp = linspace(35, 34, 5000); % Design 3
%chirp = linspace(11, 5, 5000);
%chirp = linspace(27.5,30,5000); % Design 4
%chirp = linspace(26.5, 28, 5000);
chirp = linspace(27.5,34.5,5000);
%gratingProfile = 1e-6*ones(length(zSpace),length(gratings))*diag(gratings);
gratingProfile = 1e-6*chirp;
zSpace = zSpace-zSpace(1);

% running the code 
% numSteps = 100;
% for ii = 1:length(gratings)
%     [t,Ats,totalWs,Aws]=ssfmv2(t,transpose(pump),center_wl,alpha,ldas,n,L,zSpace,0.95*chi,gratingProfile(:,ii),gamma,fR,numSteps);
%     if ii == 7
%         fullProp = Aws;
%     end
%     AwsGratings(ii,:)=Aws(end,:);
% end


numSteps = 100;
[t,Ats,totalWs,Aws]=ssfmv2(t,transpose(pump),center_wl,alpha,ldas,n,L,zSpace,0.95*chi,gratingProfile,gamma,fR,numSteps);



% numSteps = 100;
% AtsAll = zeros(length(powerFactors),numSteps,N);
% AwsPowers_spp = zeros(length(powerFactors),N);
% 
% for ii = 1:length(powerFactors)
%     [t,Ats,totalWs,Aws]=ssfmv2(t,transpose(pump(:,ii)),center_wl,alpha,ldas,n,L,zSpace,0.95*chi,gratingProfile,gamma,fR,numSteps);
%     AwsPowers_spp(ii,:)=Aws(end,:);
%     [~,indices] = find(totalWs > 0);
%     ldaMIR = c*1e-3./(totalWs(indices)/(2*pi));
%     ldaMask = ldaMIR > 2700;
%     AwsPowers_filt(ii,:) = AwsPowers_spp(ii,indices);
%     PowerIn(ii) = 250*powerFactors(ii).^2; %Power in mW
%     PowerOut(ii) = PowerIn(ii)*trapz(abs(AwsPowers_filt(ii,ldaMask)).^2)/(trapz(abs(AwsPowers_filt(ii,:)).^2));
%     AtsAll(ii,:,:) = Ats;
% end
%     
    
%plotting only the positive frequencies
% [~,indices] = find(totalWs > 0);
% % figure; % propagation in 3-d
%  ldaMIR = c*1e-3./(totalWs(indices)/(2*pi));
 
% logAws = 10*log10(abs(Aws(:,indices)).^2);
% logAws(logAws < -70) = -70;
% zs = linspace(zStart,zEnd,numSteps);

% surf(ldaMIR,zs/1e-3,logAws)
% view(2);
% shading flat;
% colormap(jet);
% xlim([500,5000])
% xlabel('Wavelength (nm)')
% ylabel('Distance (mm)')

% %plotting only the positive frequencies
% [~,indices] = find(totalWs > 0);
% figure; % propagation in 3-d
% ldaMIR = c*1e-3./(totalWs(indices)/(2*pi));
% logAws = 10*log10(abs(Aws(:,indices)).^2);
% logAws(logAws < -70) = -70;
% zs = linspace(zStart,zEnd,numSteps);
% 
% surf(ldaMIR,zs/1e-3,logAws)
% view(2);
% shading flat;
% colormap(jet);
% xlim([500,5000])
% xlabel('Wavelength (nm)')
% ylabel('Distance (mm)')
% 
% figure; % 2d 
% semilogy(ldaMIR, abs(Aws(end,indices)).^2)
