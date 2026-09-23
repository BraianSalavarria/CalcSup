import json
import re
import customtkinter as ctk
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
import scipy.integrate as integrate


class ParametricSurfaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Visualizador de Superficies Paramétricas")
        self.geometry("1400x850")

        try:
            self.state('zoomed')
        except Exception:
            pass

        # Layout Principal
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_visible = True

        # Almacenamiento de múltiples superficies
        self.surfaces_list = []

        self.trig_functions = [
            "arcsen", "arcsin", "arccos", "arctan",
            "sen", "sin", "cos", "tan", "sqrt", "exp", "log"
        ]

        self.colormaps = ["viridis", "plasma", "inferno", "magma", "coolwarm", "ocean", "copper"]

        # ----------------------------------------------------
        # Panel Izquierdo Organizado con Pestañas
        # ----------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=10)
        self.sidebar_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Título Principal
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Ecuaciones Paramétricas", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.logo_label.pack(pady=(12, 5), padx=10)

        # TabView (Menú de Pestañas Organizado)
        self.tabview = ctk.CTkTabview(self.sidebar_frame, width=300)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_surf = self.tabview.add("Superficies")
        self.tab_tangent = self.tabview.add("Tangente / Normal")
        self.tab_area = self.tabview.add("Área")
        self.tab_file = self.tabview.add("Archivo")

        # ====================================================
        # PESTAÑA 1: SUPERFICIES
        # ====================================================
        self.entry_x = self._create_input_field(self.tab_surf, "x(u, v):", "u . cos(v)")
        self.entry_y = self._create_input_field(self.tab_surf, "y(u, v):", "u . sen(v)")
        self.entry_z = self._create_input_field(self.tab_surf, "z(u, v):", "u^2")

        # Rango u
        self._add_section_header(self.tab_surf, "Rango Parámetro u")
        self.frame_u = ctk.CTkFrame(self.tab_surf, fg_color="transparent")
        self.frame_u.pack(fill="x", padx=5, pady=2)
        self.u_min = self._create_range_entry(self.frame_u, "u_min:", "0")
        self.u_max = self._create_range_entry(self.frame_u, "u_max:", "2")

        # Rango v
        self._add_section_header(self.tab_surf, "Rango Parámetro v")
        self.frame_v = ctk.CTkFrame(self.tab_surf, fg_color="transparent")
        self.frame_v.pack(fill="x", padx=5, pady=2)
        self.v_min = self._create_range_entry(self.frame_v, "v_min:", "0")
        self.v_max = self._create_range_entry(self.frame_v, "v_max:", "2.pi")

        # Malla y Paleta
        self._add_section_header(self.tab_surf, "Resolución y Estilo")
        self.resolution_entry = self._create_input_field(self.tab_surf, "Puntos (Resolución):", "40", enable_autocomplete=False)
        
        label_cmap = ctk.CTkLabel(self.tab_surf, text="Mapa de Color:", anchor="w")
        label_cmap.pack(fill="x", padx=10, pady=(2, 0))
        self.cmap_option = ctk.CTkOptionMenu(self.tab_surf, values=self.colormaps)
        self.cmap_option.set("viridis")
        self.cmap_option.pack(fill="x", padx=10, pady=(0, 5))

        # Botones de Acción de Superficie
        self.plot_button = ctk.CTkButton(
            self.tab_surf, 
            text="Graficar Superficie Principal", 
            command=self.plot_primary,
            font=ctk.CTkFont(weight="bold")
        )
        self.plot_button.pack(pady=8, padx=10, fill="x")

        self.add_surf_btn = ctk.CTkButton(
            self.tab_surf, 
            text="+ Agregar Nueva Superficie", 
            fg_color="#1098ad", hover_color="#0b7285",
            command=self.add_surface
        )
        self.add_surf_btn.pack(pady=3, padx=10, fill="x")

        # Sección para Eliminar Superficies
        self._add_section_header(self.tab_surf, "Gestionar Superficies Activas")
        self.delete_option = ctk.CTkOptionMenu(self.tab_surf, values=["Superficie 1"])
        self.delete_option.pack(fill="x", padx=10, pady=2)

        self.remove_surf_btn = ctk.CTkButton(
            self.tab_surf, 
            text="Eliminar Superficie Seleccionada", 
            fg_color="#e03131", hover_color="#c92a2a",
            command=self.remove_surface
        )
        self.remove_surf_btn.pack(pady=5, padx=10, fill="x")

        # ====================================================
        # PESTAÑA 2: PLANO TANGENTE Y VECTOR NORMAL
        # ====================================================
        self.switch_normal = ctk.CTkSwitch(self.tab_tangent, text="Mostrar Vector Normal")
        self.switch_normal.pack(anchor="w", padx=10, pady=8)
        self.switch_normal.select()

        self.switch_tangent = ctk.CTkSwitch(self.tab_tangent, text="Mostrar Plano Tangente")
        self.switch_tangent.pack(anchor="w", padx=10, pady=8)
        self.switch_tangent.select()

        self._add_section_header(self.tab_tangent, "Evaluación por Parámetros (u, v)")
        self.frame_point_uv = ctk.CTkFrame(self.tab_tangent, fg_color="transparent")
        self.frame_point_uv.pack(fill="x", padx=5, pady=2)
        self.point_u0 = self._create_range_entry(self.frame_point_uv, "u0:", "1")
        self.point_v0 = self._create_range_entry(self.frame_point_uv, "v0:", "pi/4")

        # Etiqueta de Resultados (Producto Vectorial y Ecuación)
        self.tangent_info_label = ctk.CTkLabel(
            self.tab_tangent, 
            text="Producto Vectorial ru x rv:\n-\n\nEcuación Plano Tangente:\n-", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#1864ab",
            justify="left",
            wraplength=260
        )
        self.tangent_info_label.pack(pady=15, padx=10, fill="x")

        self.calc_tangent_btn = ctk.CTkButton(
            self.tab_tangent, 
            text="Calcular Plano y Normal", 
            command=self.plot_surface
        )
        self.calc_tangent_btn.pack(pady=5, padx=10, fill="x")

        # ====================================================
        # PESTAÑA 3: ÁREA DE LA SUPERFICIE
        # ====================================================
        self._add_section_header(self.tab_area, "Cálculo de Área (Integral Doble)")
        self.area_info_label = ctk.CTkLabel(
            self.tab_area, 
            text="Área de la superficie:\n-", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#2b8a3e",
            justify="left",
            wraplength=260
        )
        self.area_info_label.pack(pady=15, padx=10, fill="x")

        self.calc_area_btn = ctk.CTkButton(
            self.tab_area, 
            text="Calcular Área de Superficie", 
            fg_color="#2b8a3e", hover_color="#216a30",
            command=self.compute_surface_area
        )
        self.calc_area_btn.pack(pady=5, padx=10, fill="x")

        # ====================================================
        # PESTAÑA 4: ARCHIVOS Y EXPORTACIÓN
        # ====================================================
        self.save_img_button = ctk.CTkButton(
            self.tab_file, 
            text="Exportar Imagen 3D", 
            fg_color="#2b8a3e", hover_color="#216a30",
            command=self.export_image
        )
        self.save_img_button.pack(pady=10, padx=10, fill="x")

        self.save_file_button = ctk.CTkButton(
            self.tab_file, text="Guardar Proyecto (.json)", 
            fg_color="#3b5bdb", hover_color="#2b44ad",
            command=self.save_project
        )
        self.save_file_button.pack(pady=5, padx=10, fill="x")

        self.load_file_button = ctk.CTkButton(
            self.tab_file, text="Abrir Proyecto (.json)", 
            fg_color="#3b5bdb", hover_color="#2b44ad",
            command=self.load_project
        )
        self.load_file_button.pack(pady=5, padx=10, fill="x")

        # ----------------------------------------------------
        # SECCIÓN INFERIOR DE ERRORES / MENSAJES DE ESTADO
        # ----------------------------------------------------
        self._add_section_header(self.sidebar_frame, "Log de Estado / Errores")
        self.status_box = ctk.CTkTextbox(self.sidebar_frame, height=75, text_color="red")
        self.status_box.pack(fill="x", padx=10, pady=(0, 10))

        # ----------------------------------------------------
        # Panel Derecho: Área de Dibujo Maximizada
        # ----------------------------------------------------
        self.plot_frame = ctk.CTkFrame(self, corner_radius=10)
        self.plot_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.fig = Figure(figsize=(10, 10), dpi=100)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Botón Flotante para Ocultar Panel
        self.toggle_btn = ctk.CTkButton(
            self.plot_frame, 
            text="◀ Ocultar Panel", 
            width=110, height=28,
            fg_color="#495057", hover_color="#343a40",
            command=self.toggle_sidebar
        )
        self.toggle_btn.place(x=10, y=10)
        self.toggle_btn.lift()

        # Eventos Zoom
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.bind("<MouseWheel>", self._on_mouse_wheel)
        canvas_widget.bind("<Button-4>", self._on_mouse_wheel)  
        canvas_widget.bind("<Button-5>", self._on_mouse_wheel)  

        self.bind("<Control-plus>", lambda e: self._zoom(0.85))
        self.bind("<Control-KP_Add>", lambda e: self._zoom(0.85))
        self.bind("<Control-minus>", lambda e: self._zoom(1.15))
        self.bind("<Control-KP_Subtract>", lambda e: self._zoom(1.15))

        # Inicializar
        self.plot_primary()

    def show_log(self, text, is_error=True):
        """Muestra mensajes o errores en el cuadro dedicado."""
        self.status_box.configure(state="normal")
        self.status_box.delete("1.0", ctk.END)
        self.status_box.configure(text_color="red" if is_error else "green")
        self.status_box.insert(ctk.END, str(text))
        self.status_box.configure(state="disabled")

    # ----------------------------------------------------
    # Lógica de Múltiples Superficies
    # ----------------------------------------------------
    def plot_primary(self):
        surf = self._get_current_input_data()
        self.surfaces_list = [surf]
        self._update_delete_dropdown()
        self.plot_surface()

    def add_surface(self):
        surf = self._get_current_input_data()
        self.surfaces_list.append(surf)
        self._update_delete_dropdown()
        self.plot_surface()

    def remove_surface(self):
        selected = self.delete_option.get()
        if not selected or not self.surfaces_list:
            return
        
        try:
            index = int(selected.split(" ")[1]) - 1
            if 0 <= index < len(self.surfaces_list):
                self.surfaces_list.pop(index)
                self._update_delete_dropdown()
                self.plot_surface()
        except Exception:
            pass

    def _get_current_input_data(self):
        return {
            "x": self.entry_x.get(),
            "y": self.entry_y.get(),
            "z": self.entry_z.get(),
            "u_min": self.u_min.get(),
            "u_max": self.u_max.get(),
            "v_min": self.v_min.get(),
            "v_max": self.v_max.get(),
            "cmap": self.cmap_option.get()
        }

    def _update_delete_dropdown(self):
        options = [f"Superficie {i+1}" for i in range(len(self.surfaces_list))]
        if not options:
            options = ["Ninguna"]
        self.delete_option.configure(values=options)
        self.delete_option.set(options[0])

    # ----------------------------------------------------
    # Ocultar / Mostrar Panel Lateral
    # ----------------------------------------------------
    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_frame.grid_remove()
            self.toggle_btn.configure(text="▶ Mostrar Panel")
            self.sidebar_visible = False
        else:
            self.sidebar_frame.grid()
            self.toggle_btn.configure(text="◀ Ocultar Panel")
            self.sidebar_visible = True
        self.canvas.draw_idle()

    # ----------------------------------------------------
    # Control de Zoom
    # ----------------------------------------------------
    def _zoom(self, factor):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        self.ax.set_xlim([x * factor for x in xlim])
        self.ax.set_ylim([y * factor for y in ylim])
        self.ax.set_zlim([z * factor for z in zlim])
        self.canvas.draw_idle()

    def _on_mouse_wheel(self, event):
        if event.delta > 0 or event.num == 4:
            self._zoom(0.88)
        elif event.delta < 0 or event.num == 5:
            self._zoom(1.12)

    # ----------------------------------------------------
    # Autocompletado y Construcción UI
    # ----------------------------------------------------
    def _on_key_release(self, event, entry_widget):
        if event.keysym in ["BackSpace", "Delete", "Left", "Right", "Up", "Down", "Return", "Tab", "Shift_L", "Shift_R"]:
            return

        cursor_pos = entry_widget.index(ctk.INSERT)
        text_before_cursor = entry_widget.get()[:cursor_pos]

        for func in self.trig_functions:
            if text_before_cursor.endswith(func) and not entry_widget.get()[cursor_pos:].startswith("("):
                entry_widget.insert(cursor_pos, "()")
                entry_widget.icursor(cursor_pos + 1)
                break

    def _create_input_field(self, parent, label_text, default_value, enable_autocomplete=True):
        label = ctk.CTkLabel(parent, text=label_text, anchor="w")
        label.pack(fill="x", padx=10, pady=(3, 0))
        entry = ctk.CTkEntry(parent)
        entry.insert(0, default_value)
        entry.pack(fill="x", padx=10, pady=(0, 3))

        if enable_autocomplete:
            entry.bind("<KeyRelease>", lambda event: self._on_key_release(event, entry))

        return entry

    def _create_range_entry(self, parent, label_text, default_value):
        sub_frame = ctk.CTkFrame(parent, fg_color="transparent")
        sub_frame.pack(side="left", expand=True, fill="x", padx=2)
        
        label = ctk.CTkLabel(sub_frame, text=label_text, anchor="w")
        label.pack(fill="x")
        entry = ctk.CTkEntry(sub_frame, width=100)
        entry.insert(0, default_value)
        entry.pack(fill="x")

        entry.bind("<KeyRelease>", lambda event: self._on_key_release(event, entry))

        return entry

    def _add_section_header(self, parent, title):
        header = ctk.CTkLabel(
            parent, 
            text=title, 
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        )
        header.pack(fill="x", padx=10, pady=(8, 2))

    # ----------------------------------------------------
    # Procesamiento Matemático
    # ----------------------------------------------------
    def _preprocess_expr(self, expr: str) -> str:
        expr = expr.replace("^", "**")
        expr = re.sub(r'\bsen\b', 'sin', expr)
        expr = re.sub(r'(?<=[a-zA-Z0-9\)])\s*\.\s*(?=[a-zA-Z\(])', '*', expr)
        expr = re.sub(r'(\d)\s*([a-zA-Z\(])', r'\1*\2', expr)
        return expr

    def _eval_expr(self, expr_str, local_dict):
        processed_str = self._preprocess_expr(expr_str)
        u_sym, v_sym = sp.symbols('u v')
        
        context = {
            'u': u_sym, 'v': v_sym, 
            'pi': sp.pi, 'e': sp.E, 'E': sp.E,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'sqrt': sp.sqrt
        }
        
        parsed_expr = sp.sympify(processed_str, locals=context)
        
        if any(s in parsed_expr.free_symbols for s in (u_sym, v_sym)):
            func = sp.lambdify((u_sym, v_sym), parsed_expr, modules=['numpy'])
            u_val = local_dict.get('u', 0)
            v_val = local_dict.get('v', 0)
            return func(u_val, v_val)
        else:
            return float(parsed_expr.evalf())

    # ----------------------------------------------------
    # Cálculo del Plano Tangente, Vector Normal y Área
    # ----------------------------------------------------
    def _compute_normal_and_tangent(self, surf, u0_val, v0_val):
        u_sym, v_sym = sp.symbols('u v')
        
        x_expr = sp.sympify(self._preprocess_expr(surf['x']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
        y_expr = sp.sympify(self._preprocess_expr(surf['y']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
        z_expr = sp.sympify(self._preprocess_expr(surf['z']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})

        # Derivadas ru y rv
        ru = [sp.diff(x_expr, u_sym), sp.diff(y_expr, u_sym), sp.diff(z_expr, u_sym)]
        rv = [sp.diff(x_expr, v_sym), sp.diff(y_expr, v_sym), sp.diff(z_expr, v_sym)]

        # Producto Vectorial N = ru x rv
        N_x = ru[1]*rv[2] - ru[2]*rv[1]
        N_y = ru[2]*rv[0] - ru[0]*rv[2]
        N_z = ru[0]*rv[1] - ru[1]*rv[0]

        subs_dict = {u_sym: u0_val, v_sym: v0_val, sp.pi: np.pi, sp.E: np.e}
        
        p0 = np.array([
            float(x_expr.subs(subs_dict).evalf()),
            float(y_expr.subs(subs_dict).evalf()),
            float(z_expr.subs(subs_dict).evalf())
        ])

        N_vec = np.array([
            float(N_x.subs(subs_dict).evalf()),
            float(N_y.subs(subs_dict).evalf()),
            float(N_z.subs(subs_dict).evalf())
        ])

        norm_val = np.linalg.norm(N_vec)
        N_unit = N_vec / norm_val if norm_val != 0 else N_vec

        A, B, C = N_vec
        D = -(A*p0[0] + B*p0[1] + C*p0[2])
        plane_eq = f"{A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0"

        return p0, N_unit, N_vec, plane_eq

    def compute_surface_area(self):
        try:
            if not self.surfaces_list:
                self.show_log("No hay ninguna superficie activa para calcular el área.", is_error=True)
                return
            
            last_surf = self.surfaces_list[-1]
            u_sym, v_sym = sp.symbols('u v')
            
            x_expr = sp.sympify(self._preprocess_expr(last_surf['x']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
            y_expr = sp.sympify(self._preprocess_expr(last_surf['y']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
            z_expr = sp.sympify(self._preprocess_expr(last_surf['z']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})

            # Derivadas parciales r_u y r_v
            ru = [sp.diff(x_expr, u_sym), sp.diff(y_expr, u_sym), sp.diff(z_expr, u_sym)]
            rv = [sp.diff(x_expr, v_sym), sp.diff(y_expr, v_sym), sp.diff(z_expr, v_sym)]

            # Producto Vectorial N = r_u x r_v
            N_x = ru[1]*rv[2] - ru[2]*rv[1]
            N_y = ru[2]*rv[0] - ru[0]*rv[2]
            N_z = ru[0]*rv[1] - ru[1]*rv[0]

            # Magnitud del vector normal: ||r_u x r_v||
            magnitude_expr = sp.sqrt(N_x**2 + N_y**2 + N_z**2)

            # Evaluar límites de integración para u y v
            u_min_val = float(self._eval_expr(last_surf['u_min'], {}))
            u_max_val = float(self._eval_expr(last_surf['u_max'], {}))
            v_min_val = float(self._eval_expr(last_surf['v_min'], {}))
            v_max_val = float(self._eval_expr(last_surf['v_max'], {}))

            # Convertir la expresión simbólica a una función numérica optimizada con numpy
            f_area = sp.lambdify((u_sym, v_sym), magnitude_expr, modules=['numpy'])

            # Resolver la integral doble mediante cuadratura numérica de SciPy
            area_val, _ = integrate.dblquad(
                lambda v_val, u_val: float(f_area(u_val, v_val)),
                u_min_val, u_max_val,
                lambda u: v_min_val,
                lambda u: v_max_val
            )

            # Mostrar el resultado en la interfaz
            self.area_info_label.configure(
                text=f"Área de la superficie:\nA = {area_val:.4f} unidades²"
            )
            self.show_log("Área calculada exitosamente.", is_error=False)

        except Exception as e:
            self.show_log(f"Error al calcular el área:\n{str(e)}", is_error=True)

    # ----------------------------------------------------
    # Graficado 3D Interactivo
    # ----------------------------------------------------
    def plot_surface(self):
        self.show_log("Escena renderizada sin errores.", is_error=False)
        try:
            if not self.surfaces_list:
                self.ax.clear()
                self.canvas.draw()
                return

            elev, azim = self.ax.elev, self.ax.azim
            self.ax.clear()

            # Estilo GeoGebra
            self.ax.set_facecolor('white')
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('white')
            self.ax.yaxis.pane.set_edgecolor('white')
            self.ax.zaxis.pane.set_edgecolor('white')
            self.ax.grid(False)

            all_X, all_Y, all_Z = [], [], []
            n_points = int(self.resolution_entry.get())

            # 1. Renderizar cada superficie
            for surf in self.surfaces_list:
                u_min_val = self._eval_expr(surf['u_min'], {})
                u_max_val = self._eval_expr(surf['u_max'], {})
                v_min_val = self._eval_expr(surf['v_min'], {})
                v_max_val = self._eval_expr(surf['v_max'], {})

                u_vals = np.linspace(u_min_val, u_max_val, n_points)
                v_vals = np.linspace(v_min_val, v_max_val, n_points)
                u, v = np.meshgrid(u_vals, v_vals)

                X = self._eval_expr(surf['x'], {"u": u, "v": v})
                Y = self._eval_expr(surf['y'], {"u": u, "v": v})
                Z = self._eval_expr(surf['z'], {"u": u, "v": v})

                if np.isscalar(X): X = np.full_like(u, X)
                if np.isscalar(Y): Y = np.full_like(u, Y)
                if np.isscalar(Z): Z = np.full_like(u, Z)

                all_X.append(X); all_Y.append(Y); all_Z.append(Z)

                self.ax.plot_surface(
                    X, Y, Z, 
                    cmap=surf.get('cmap', 'viridis'), 
                    edgecolor="none", 
                    alpha=0.75,
                    antialiased=True
                )

            # 2. Renderizar Punto, Vector Normal y Plano Tangente
            if self.switch_normal.get() or self.switch_tangent.get():
                last_surf = self.surfaces_list[-1]
                u0_val = self._eval_expr(self.point_u0.get(), {})
                v0_val = self._eval_expr(self.point_v0.get(), {})

                p0, N_unit, N_raw, plane_eq = self._compute_normal_and_tangent(last_surf, u0_val, v0_val)

                # Actualizar información detallada en el menú lateral
                self.tangent_info_label.configure(
                    text=f"Producto Vectorial ru x rv:\n({N_raw[0]:.2f}, {N_raw[1]:.2f}, {N_raw[2]:.2f})\n\nEcuación Plano Tangente:\n{plane_eq}"
                )

                # Graficar Punto P0 con sus coordenadas
                self.ax.scatter([p0[0]], [p0[1]], [p0[2]], color='black', s=60, zorder=10)
                self.ax.text(p0[0], p0[1], p0[2]*1.05, f"P({p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f})", color='black', weight='bold')

                # Vector Normal N y sus componentes sobre el gráfico
                if self.switch_normal.get():
                    scale = 2.0
                    self.ax.quiver(
                        p0[0], p0[1], p0[2],
                        N_unit[0]*scale, N_unit[1]*scale, N_unit[2]*scale,
                        color='magenta', linewidth=3, arrow_length_ratio=0.2
                    )
                    self.ax.text(
                        p0[0] + N_unit[0]*scale*1.1, 
                        p0[1] + N_unit[1]*scale*1.1, 
                        p0[2] + N_unit[2]*scale*1.1, 
                        f"N({N_raw[0]:.2f}, {N_raw[1]:.2f}, {N_raw[2]:.2f})", 
                        color='magenta', fontsize=10, weight='bold'
                    )

                # Plano Tangente
                if self.switch_tangent.get() and np.linalg.norm(N_raw) != 0:
                    d = -np.dot(N_raw, p0)
                    tx = np.linspace(p0[0]-1.5, p0[0]+1.5, 10)
                    ty = np.linspace(p0[1]-1.5, p0[1]+1.5, 10)
                    TX, TY = np.meshgrid(tx, ty)
                    if N_raw[2] != 0:
                        TZ = (-N_raw[0]*TX - N_raw[1]*TY - d) / N_raw[2]
                        self.ax.plot_surface(TX, TY, TZ, color='orange', alpha=0.4, shade=False)

            # 3. Límites Simétricos
            concat_X = np.concatenate([x.flatten() for x in all_X])
            concat_Y = np.concatenate([y.flatten() for y in all_Y])
            concat_Z = np.concatenate([z.flatten() for z in all_Z])

            max_range = max(
                np.max(np.abs(concat_X)), 
                np.max(np.abs(concat_Y)), 
                np.max(np.abs(concat_Z)),
                3.0
            )
            limit = float(np.ceil(max_range * 1.4))

            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_zlim(-limit, limit)

            # 4. Cuadrícula Plano Z = 0
            grid_steps = np.linspace(-limit, limit, 21)
            gx, gy = np.meshgrid(grid_steps, grid_steps)
            gz = np.zeros_like(gx)
            self.ax.plot_wireframe(gx, gy, gz, color='gray', alpha=0.25, linewidth=0.7)

            # 5. Ejes Vectoriales
            self.ax.quiver(-limit, 0, 0, 2*limit, 0, 0, color='red', arrow_length_ratio=0.03, linewidth=2)
            self.ax.quiver(0, -limit, 0, 0, 2*limit, 0, color='green', arrow_length_ratio=0.03, linewidth=2)
            self.ax.quiver(0, 0, -limit, 0, 0, 2*limit, color='blue', arrow_length_ratio=0.03, linewidth=2)

            ticks = np.arange(-int(limit)+1, int(limit), 2)
            ticks = ticks[ticks != 0]

            for t in ticks:
                self.ax.text(t, 0, 0, f"{t}", color='darkred', fontsize=8, ha='center', weight='bold')
                self.ax.text(0, t, 0, f"{t}", color='darkgreen', fontsize=8, ha='center', weight='bold')
                self.ax.text(0, 0, t, f"{t}", color='darkblue', fontsize=8, ha='center', weight='bold')

            self.ax.text(limit*1.05, 0, 0, "X", color='red', fontsize=12, weight='bold')
            self.ax.text(0, limit*1.05, 0, "Y", color='green', fontsize=12, weight='bold')
            self.ax.text(0, 0, limit*1.05, "Z", color='blue', fontsize=12, weight='bold')

            self.ax.set_axis_off()

            if elev is not None and azim is not None:
                self.ax.view_init(elev=elev, azim=azim)

            self.canvas.draw()

        except Exception as e:
            self.show_log(f"Error en la entrada o cálculo:\n{str(e)}", is_error=True)

    # ----------------------------------------------------
    # Exportación / Proyectos
    # ----------------------------------------------------
    def export_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.show_log("Imagen guardada correctamente.", is_error=False)

    def save_project(self):
        data = {
            "surfaces": self.surfaces_list,
            "u0": self.point_u0.get(),
            "v0": self.point_v0.get(),
            "resolution": self.resolution_entry.get()
        }
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON File", "*.json")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            self.show_log("Proyecto guardado con éxito.", is_error=False)

    def load_project(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON File", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "surfaces" in data:
                    self.surfaces_list = data["surfaces"]
                    self._update_delete_dropdown()

                if "u0" in data: self.point_u0.delete(0, ctk.END); self.point_u0.insert(0, str(data["u0"]))
                if "v0" in data: self.point_v0.delete(0, ctk.END); self.point_v0.insert(0, str(data["v0"]))
                if "resolution" in data: self.resolution_entry.delete(0, ctk.END); self.resolution_entry.insert(0, str(data["resolution"]))

                self.plot_surface()
                self.show_log("Proyecto cargado con éxito.", is_error=False)
            except Exception as e:
                self.show_log(f"Error al abrir archivo:\n{str(e)}", is_error=True)


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ParametricSurfaceApp()
    app.mainloop()