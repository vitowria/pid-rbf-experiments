function dx = robot_dynamics_v0(t, x, params)
    % Esta função calcula as derivadas do estado do sistema para o solver ODE.

    %% 1. Desempacotar Vetor de Estado e Parâmetros
    % O vetor de estado 'x' contém a posição, velocidade, erro integral e pesos da NN.
    n = params.n; % Número de juntas (2)
    N = params.N; % Número de neurônios da RBF

    q = x(1:n);
    q_dot = x(n+1:2*n);
    e_int = x(2*n+1:3*n);
    Omega_hat_vec = x(3*n+1:end);
    Omega_hat = reshape(Omega_hat_vec, N, n); % Pesos da NN [N x n]

    %% 2. Trajetória Desejada (Senoide Simples)
    qd = [sin(t); cos(t)];
    qd_dot = [cos(t); -sin(t)];
    qd_ddot = [-sin(t); -cos(t)];

    %% 3. Cálculo dos Erros do PID
    e = qd - q; % Erro de Posição 
    e_dot = qd_dot - q_dot; % Erro de Velocidade

    % Cálculo do Erro Generalizado (Xi) 
    Xi = 2 * params.k0 * e + params.k0^2 * e_int + e_dot;

    %% 4. Estrutura da RBF NN com Lei de Adaptação
    % a. Entrada da Rede Neural (Z)
    % Conforme o artigo, a entrada Z pode conter vários sinais. 
    % Uma escolha prática para implementação é usar os sinais de erro.
    Z = [e; e_dot; Xi];

    % b. Cálculo da Saída da Camada RBF (h)
    h = zeros(N, 1);
    for i = 1:N
        % Função de base Gaussiana 
        h(i) = exp(-norm(Z - params.mu(:, i))^2 / (2 * params.rho^2));
    end

    % c. Cálculo da Derivada dos Pesos da NN (Lei Adaptativa)
    % A lei adaptativa é definida pela Eq. (8) do artigo. 
    Omega_hat_dot = zeros(N, n);
    for i=1:n
        Omega_hat_dot(:,i) = params.Gamma * (params.alpha * (Xi(i)^2) * h - params.varpi * Omega_hat(:,i));
    end

    %% 5. Implementação do Controlador PID Adaptativo
    % O torque é τ = K_D * Ξ 
    % Onde K_D é composto por uma parte constante e uma adaptativa
    % κ_D(t) = α * Ω̂ᵀ * h 

    kappa_D = zeros(n, 1);
    for i=1:n
       kappa_D(i) = params.alpha * Omega_hat(:,i)' * h;
    end
    
    % Ganho total (constante + adaptativo)
    K_D = params.kd + kappa_D;

    % Torque de controle final (um para cada junta)
    tau = K_D .* Xi;

    %% 6. Dinâmica do Manipulador Robótico
    % Matrizes de dinâmica M, C, G para o robô de 2 juntas 
    p1 = params.p1; p2 = params.p2; p3 = params.p3; p4 = params.p4; p5 = params.p5; g = 9.81;
    q1 = q(1); q2 = q(2); q1d = q_dot(1); q2d = q_dot(2);
    
    M = [p1 + p2 + 2*p3*cos(q2), p2 + p3*cos(q2);
         p2 + p3*cos(q2),          p2];
    
    C = [-p3*sin(q2)*q2d,  -p3*sin(q2)*(q1d + q2d);
          p3*sin(q2)*q1d,   0];
      
    G = [p4*g*cos(q1) + p5*g*cos(q1+q2);
         p5*g*cos(q1+q2)];
     
    % Para esta semana, o distúrbio 'd' é zero.
    d = [0; 0];
    
    % Aceleração do Robô (q_ddot)
    q_ddot = M \ (tau + d - C * q_dot - G);

    %% 7. Montar o vetor de derivadas 'dx' para o solver
    dx = zeros(size(x));
    dx(1:n) = q_dot;          % Derivada da posição é a velocidade
    dx(n+1:2*n) = q_ddot;       % Derivada da velocidade é a aceleração
    dx(2*n+1:3*n) = e;            % Derivada do erro integral é o erro
    dx(3*n+1:end) = reshape(Omega_hat_dot, N * n, 1); % Derivada dos pesos da NN
end