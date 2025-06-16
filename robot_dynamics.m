function dx = robot_dynamics(t, x, params)
%--------------------------------------------------------------------------
% FUNÇÃO DE DINÂMICA DO SISTEMA
%--------------------------------------------------------------------------
% Esta função calcula as derivadas do estado do sistema (posição,
% velocidade, erro integral e pesos da NN) para o solver ODE.
% É aqui que a lógica do controlador e a física do robô são implementadas.
%--------------------------------------------------------------------------

%% 1. Desempacotar Vetor de Estado e Parâmetros
n = params.n;
N = params.N;

q = x(1:n);
q_dot = x(n+1:2*n);
e_int = x(2*n+1:3*n);
Omega_hat_vec = x(3*n+1:end);
Omega_hat = reshape(Omega_hat_vec, N, n);

%% 2. Trajetória Desejada
% Conforme Seção 5 do artigo
qd = [cos(t); -cos(t)];
qd_dot = [-sin(t); sin(t)];
qd_ddot = [-cos(t); cos(t)];

%% 3. Cálculo dos Erros
e = qd - q;
e_dot = qd_dot - q_dot;
Xi = 2 * params.k0 * e + params.k0^2 * e_int + e_dot;

%% 4. Lógica da Rede Neural RBF e Lei de Adaptação
% a. Entrada da Rede Neural (Z)
Z = [e; e_dot; Xi];

% b. Cálculo da Saída da Camada RBF (h)
h = zeros(N, 1);
for i = 1:N
    h(i) = exp(-norm(Z - params.mu(:, i))^2 / (2 * params.rho^2));
end

% c. Cálculo da Derivada dos Pesos da NN (Lei Adaptativa Eq. 8)
Omega_hat_dot = zeros(N, n);
for i=1:n
    Omega_hat_dot(:,i) = params.Gamma * (params.alpha * (Xi(i)^2) * h - params.varpi * Omega_hat(:,i));
end

%% 5. Cálculo do Torque de Controle (Controlador com BLF)
% Parte adaptativa do ganho
kappa_D = params.alpha * (Omega_hat' * h);

% Termo de ganho da Função de Barreira de Lyapunov (BLF)
K_dblf = params.kb ./ (params.C^2 - Xi.^2 + 1e-9);

% Ganho total (constante + adaptativo + barreira)
K_Dblf_total = params.kd + kappa_D + K_dblf;

% Torque de controle final
tau = K_Dblf_total .* Xi;

%% 6. Dinâmica do Manipulador Robótico (Modelo Físico)
% Matrizes M, C, G para o robô de 2 juntas (Seção 5)
p1 = params.p1; p2 = params.p2; p3 = params.p3; p4 = params.p4; p5 = params.p5; g = params.g;
q1 = q(1); q2 = q(2); q1d = q_dot(1); q2d = q_dot(2);

M = [p1 + 2*p3*cos(q2), p2 + p3*cos(q2);
     p2 + p3*cos(q2),    p2];

C = [-p3*sin(q2)*q2d,  -p3*sin(q2)*(q1d + q2d);
      p3*sin(q2)*q1d,   0];
  
G = [p4*g*cos(q1) + p5*g*cos(q1+q2);
     p5*g*cos(q1+q2)];
 
% Distúrbio (conforme Seção 5 do artigo)
d = [0.4*sin(t); -3*cos(t)];

% Aceleração do Robô
q_ddot = M \ (tau + d - C * q_dot - G);

%% 7. Montar o vetor de derivadas 'dx' para o solver
dx = zeros(size(x));
dx(1:n) = q_dot;
dx(n+1:2*n) = q_ddot;
dx(2*n+1:3*n) = e;
dx(3*n+1:end) = reshape(Omega_hat_dot, N * n, 1);
end