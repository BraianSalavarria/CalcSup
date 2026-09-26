import json
import re
import customtkinter as ctk
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

        # Almacenamiento de múltiples superficies y control de selección
        self.surfaces_list = []
        self.selected_surface_index = 0

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

        # TabView
        self.tabview = ctk.CTkTabview(self.sidebar_frame, width=300)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_surf = self.tabview.add("Superficies")
        self.tab_tangent = self.tabview.add("Tangente / Normal")
        self.tab_area = self.tabview.add("Área")
        self.tab_file = self.tabview.add("Archivo")

        # ====================================================
        # PESTAÑA 1: SUPERFICIES
        # ====================================================
        self.entry_x = self._create_input_field(self.tab_surf, "x(u, v):", "e^u . cos(v)")
        self.entry_y = self._create_input_field(self.tab_surf, "y(u, v):", "e^u . sen(v)")
        self.entry_z = self._create_input_field(self.tab_surf, "z(u, v):", "u")

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

        # Botones de Acción
        self.plot_button = ctk.CTkButton(
            self.tab_surf, 
            text="Graficar como Principal única", 
            command=self.plot_primary,
            font=ctk.CTkFont(weight="bold")
        )
        self.plot_button.pack(pady=6, padx=10, fill="x")

        self.add_surf_btn = ctk.CTkButton(
            self.tab_surf, 
            text="+ Agregar Nueva Superficie", 
            fg_color="#1098ad", hover_color="#0b7285",
            command=self.add_surface
        )
        self.add_surf_btn.pack(pady=3, padx=10, fill="x")

        # ====================================================
        # SECCIÓN: GESTIONAR SUPERFICIES ACTIVAS
        # ====================================================
        self._add_section_header(self.tab_surf, "Gestionar Superficies Activas")
        
        self.select_option = ctk.CTkOptionMenu(
            self.tab_surf, 
            values=["Superficie 1"], 
            command=self.on_surface_selected
        )
        self.select_option.pack(fill="x", padx=10, pady=2)

        self.update_surf_btn = ctk.CTkButton(
            self.tab_surf, 
            text="Guardar Cambios en Seleccionada", 
            fg_color="#f59f00", hover_color="#f08c00", text_color="black",
            command=self.update_selected_surface
        )
        self.update_surf_btn.pack(pady=3, padx=10, fill="x")

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
        self.switch_normal = ctk.CTkSwitch(self.tab_tangent, text="Mostrar Vector Normal (N)")
        self.switch_normal.pack(anchor="w", padx=10, pady=4)
        self.switch_normal.select()

        self.switch_ru = ctk.CTkSwitch(self.tab_tangent, text="Mostrar Tangente ru")
        self.switch_ru.pack(anchor="w", padx=10, pady=4)
        self.switch_ru.select()

        self.switch_rv = ctk.CTkSwitch(self.tab_tangent, text="Mostrar Tangente rv")
        self.switch_rv.pack(anchor="w", padx=10, pady=4)
        self.switch_rv.select()

        self.switch_tangent = ctk.CTkSwitch(self.tab_tangent, text="Mostrar Plano Tangente")
        self.switch_tangent.pack(anchor="w", padx=10, pady=4)
        self.switch_tangent.select()

        self._add_section_header(self.tab_tangent, "Formato de Representación")
        self.vector_format_seg = ctk.CTkSegmentedButton(
            self.tab_tangent, 
            values=["Cartesiano", "Vectorial"],
            command=lambda v: self.plot_surface()
        )
        self.vector_format_seg.set("Cartesiano")
        self.vector_format_seg.pack(fill="x", padx=10, pady=4)

        self._add_section_header(self.tab_tangent, "Evaluación por Parámetros (u, v)")
        self.frame_point_uv = ctk.CTkFrame(self.tab_tangent, fg_color="transparent")
        self.frame_point_uv.pack(fill="x", padx=5, pady=2)
        self.point_u0 = self._create_range_entry(self.frame_point_uv, "u0:", "1")
        self.point_v0 = self._create_range_entry(self.frame_point_uv, "v0:", "pi/4")

        self.tangent_info_label = ctk.CTkLabel(
            self.tab_tangent, 
            text="Vector ru:\n-\n\nVector rv:\n-\n\nNormal N = ru x rv:\n-\n\nPlano Tangente:\n-", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#1864ab",
            justify="left",
            wraplength=260
        )
        self.tangent_info_label.pack(pady=10, padx=10, fill="x")

        self.calc_tangent_btn = ctk.CTkButton(
            self.tab_tangent, 
            text="Calcular Plano y Vectores", 
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
    # Formateador de Vectores
    # ----------------------------------------------------
    def _format_vector(self, vec):
        fmt = self.vector_format_seg.get()
        vx, vy, vz = vec[0], vec[1], vec[2]
        if fmt == "Vectorial":
            return f"{vx:.2f}i + {vy:.2f}j + {vz:.2f}k"
        else:
            return f"({vx:.2f}; {vy:.2f}; {vz:.2f})"

    # ----------------------------------------------------
    # Lógica de Múltiples Superficies y Selección
    # ----------------------------------------------------
    def plot_primary(self):
        surf = self._get_current_input_data()
        self.surfaces_list = [surf]
        self.selected_surface_index = 0
        self._update_selection_dropdown()
        self.plot_surface()

    def add_surface(self):
        surf = self._get_current_input_data()
        self.surfaces_list.append(surf)
        self.selected_surface_index = len(self.surfaces_list) - 1
        self._update_selection_dropdown()
        self.plot_surface()

    def update_selected_surface(self):
        if not self.surfaces_list:
            return
        idx = self.selected_surface_index
        if 0 <= idx < len(self.surfaces_list):
            self.surfaces_list[idx] = self._get_current_input_data()
            self.plot_surface()
            self.show_log(f"Superficie {idx+1} actualizada correctamente.", is_error=False)

    def remove_surface(self):
        selected = self.select_option.get()
        if not selected or not self.surfaces_list or selected == "Ninguna":
            return
        
        try:
            index = int(selected.split(" ")[1]) - 1
            if 0 <= index < len(self.surfaces_list):
                self.surfaces_list.pop(index)
                self.selected_surface_index = max(0, index - 1)
                self._update_selection_dropdown()
                self.plot_surface()
        except Exception:
            pass

    def on_surface_selected(self, choice):
        if not self.surfaces_list or choice == "Ninguna":
            return
        try:
            index = int(choice.split(" ")[1]) - 1
            if 0 <= index < len(self.surfaces_list):
                self.selected_surface_index = index
                surf = self.surfaces_list[index]
                self._load_surface_into_entries(surf)
                self.plot_single_surface(index)
        except Exception:
            pass

    def _load_surface_into_entries(self, surf):
        """Carga los datos de un diccionario de superficie en los controles UI."""
        self.entry_x.delete(0, ctk.END); self.entry_x.insert(0, surf['x'])
        self.entry_y.delete(0, ctk.END); self.entry_y.insert(0, surf['y'])
        self.entry_z.delete(0, ctk.END); self.entry_z.insert(0, surf['z'])
        self.u_min.delete(0, ctk.END); self.u_min.insert(0, surf['u_min'])
        self.u_max.delete(0, ctk.END); self.u_max.insert(0, surf['u_max'])
        self.v_min.delete(0, ctk.END); self.v_min.insert(0, surf['v_min'])
        self.v_max.delete(0, ctk.END); self.v_max.insert(0, surf['v_max'])
        self.cmap_option.set(surf.get('cmap', 'viridis'))

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

    def _update_selection_dropdown(self):
        options = [f"Superficie {i+1}" for i in range(len(self.surfaces_list))]
        if not options:
            options = ["Ninguna"]
        self.select_option.configure(values=options)
        if 0 <= self.selected_surface_index < len(options):
            self.select_option.set(options[self.selected_surface_index])
        else:
            self.select_option.set(options[0])

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
    # Autocompletado y UI
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
    # Cálculo de Vectores Tangentes, Normal y Plano Tangente
    # ----------------------------------------------------
    def _compute_vectors_and_tangent(self, surf, u0_val, v0_val):
        u_sym, v_sym = sp.symbols('u v')
        
        x_expr = sp.sympify(self._preprocess_expr(surf['x']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
        y_expr = sp.sympify(self._preprocess_expr(surf['y']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
        z_expr = sp.sympify(self._preprocess_expr(surf['z']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})

        ru_expr = [sp.diff(x_expr, u_sym), sp.diff(y_expr, u_sym), sp.diff(z_expr, u_sym)]
        rv_expr = [sp.diff(x_expr, v_sym), sp.diff(y_expr, v_sym), sp.diff(z_expr, v_sym)]

        N_x = ru_expr[1]*rv_expr[2] - ru_expr[2]*rv_expr[1]
        N_y = ru_expr[2]*rv_expr[0] - ru_expr[0]*rv_expr[2]
        N_z = ru_expr[0]*rv_expr[1] - ru_expr[1]*rv_expr[0]

        subs_dict = {u_sym: u0_val, v_sym: v0_val, sp.pi: np.pi, sp.E: np.e}
        
        p0 = np.array([
            float(x_expr.subs(subs_dict).evalf()),
            float(y_expr.subs(subs_dict).evalf()),
            float(z_expr.subs(subs_dict).evalf())
        ])

        ru_vec = np.array([
            float(ru_expr[0].subs(subs_dict).evalf()),
            float(ru_expr[1].subs(subs_dict).evalf()),
            float(ru_expr[2].subs(subs_dict).evalf())
        ])

        rv_vec = np.array([
            float(rv_expr[0].subs(subs_dict).evalf()),
            float(rv_expr[1].subs(subs_dict).evalf()),
            float(rv_expr[2].subs(subs_dict).evalf())
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

        return p0, ru_vec, rv_vec, N_unit, N_vec, plane_eq

    def compute_surface_area(self):
        try:
            if not self.surfaces_list:
                self.show_log("No hay ninguna superficie activa para calcular el área.", is_error=True)
                return
            
            last_surf = self.surfaces_list[self.selected_surface_index]
            u_sym, v_sym = sp.symbols('u v')
            
            x_expr = sp.sympify(self._preprocess_expr(last_surf['x']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
            y_expr = sp.sympify(self._preprocess_expr(last_surf['y']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})
            z_expr = sp.sympify(self._preprocess_expr(last_surf['z']), locals={'u': u_sym, 'v': v_sym, 'e': sp.E, 'pi': sp.pi})

            ru = [sp.diff(x_expr, u_sym), sp.diff(y_expr, u_sym), sp.diff(z_expr, u_sym)]
            rv = [sp.diff(x_expr, v_sym), sp.diff(y_expr, v_sym), sp.diff(z_expr, v_sym)]

            N_x = ru[1]*rv[2] - ru[2]*rv[1]
            N_y = ru[2]*rv[0] - ru[0]*rv[2]
            N_z = ru[0]*rv[1] - ru[1]*rv[0]

            magnitude_expr = sp.sqrt(N_x**2 + N_y**2 + N_z**2)

            u_min_val = float(self._eval_expr(last_surf['u_min'], {}))
            u_max_val = float(self._eval_expr(last_surf['u_max'], {}))
            v_min_val = float(self._eval_expr(last_surf['v_min'], {}))
            v_max_val = float(self._eval_expr(last_surf['v_max'], {}))

            f_area = sp.lambdify((u_sym, v_sym), magnitude_expr, modules=['numpy'])

            area_val, _ = integrate.dblquad(
                lambda v_val, u_val: float(f_area(u_val, v_val)),
                u_min_val, u_max_val,
                lambda u: v_min_val,
                lambda u: v_max_val
            )

            self.area_info_label.configure(
                text=f"Área (Sup. {self.selected_surface_index+1}):\nA = {area_val:.4f} unidades²"
            )
            self.show_log("Área calculada exitosamente.", is_error=False)

        except Exception as e:
            self.show_log(f"Error al calcular el área:\n{str(e)}", is_error=True)

    # ----------------------------------------------------
    # Graficado 3D Interactivo
    # ----------------------------------------------------
    def plot_surface(self):
        self._render_surfaces_sub(self.surfaces_list)

    def plot_single_surface(self, index):
        if 0 <= index < len(self.surfaces_list):
            self._render_surfaces_sub([self.surfaces_list[index]])

    def _render_surfaces_sub(self, surfaces_to_plot):
        self.show_log("Escena renderizada sin errores.", is_error=False)
        try:
            if not surfaces_to_plot:
                self.ax.clear()
                self.canvas.draw()
                return

            elev, azim = self.ax.elev, self.ax.azim
            self.ax.clear()

            # Estilo GeoGebra de Fondo
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

            for surf in surfaces_to_plot:
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

            # Renderizar Punto, Vectores Tangentes, Normal y Plano
            if (self.switch_normal.get() or self.switch_tangent.get() or self.switch_ru.get() or self.switch_rv.get()) and self.surfaces_list:
                active_surf = self.surfaces_list[self.selected_surface_index]
                u0_val = self._eval_expr(self.point_u0.get(), {})
                v0_val = self._eval_expr(self.point_v0.get(), {})

                p0, ru_vec, rv_vec, N_unit, N_raw, plane_eq = self._compute_vectors_and_tangent(active_surf, u0_val, v0_val)

                str_ru = self._format_vector(ru_vec)
                str_rv = self._format_vector(rv_vec)
                str_N = self._format_vector(N_raw)

                self.tangent_info_label.configure(
                    text=f"Vector ru:\n{str_ru}\n\nVector rv:\n{str_rv}\n\nNormal N = ru x rv:\n{str_N}\n\nPlano Tangente:\n{plane_eq}"
                )

                # Graficar Punto P0
                self.ax.scatter([p0[0]], [p0[1]], [p0[2]], color='black', s=60, zorder=10)
                self.ax.text(p0[0], p0[1], p0[2]*1.05, f"P({p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f})", color='black', weight='bold')

                scale = 2.0

                # Vector Tangente ru
                if self.switch_ru.get():
                    norm_ru = np.linalg.norm(ru_vec)
                    ru_unit = ru_vec / norm_ru if norm_ru != 0 else ru_vec
                    self.ax.quiver(
                        p0[0], p0[1], p0[2],
                        ru_unit[0]*scale, ru_unit[1]*scale, ru_unit[2]*scale,
                        color='cyan', linewidth=2.5, arrow_length_ratio=0.15
                    )
                    self.ax.text(
                        p0[0] + ru_unit[0]*scale*1.1, p0[1] + ru_unit[1]*scale*1.1, p0[2] + ru_unit[2]*scale*1.1,
                        f"ru {str_ru}", color='cyan', fontsize=9, weight='bold'
                    )

                # Vector Tangente rv
                if self.switch_rv.get():
                    norm_rv = np.linalg.norm(rv_vec)
                    rv_unit = rv_vec / norm_rv if norm_rv != 0 else rv_vec
                    self.ax.quiver(
                        p0[0], p0[1], p0[2],
                        rv_unit[0]*scale, rv_unit[1]*scale, rv_unit[2]*scale,
                        color='gold', linewidth=2.5, arrow_length_ratio=0.15
                    )
                    self.ax.text(
                        p0[0] + rv_unit[0]*scale*1.1, p0[1] + rv_unit[1]*scale*1.1, p0[2] + rv_unit[2]*scale*1.1,
                        f"rv {str_rv}", color='darkgoldenrod', fontsize=9, weight='bold'
                    )

                # Vector Normal N
                if self.switch_normal.get():
                    self.ax.quiver(
                        p0[0], p0[1], p0[2],
                        N_unit[0]*scale, N_unit[1]*scale, N_unit[2]*scale,
                        color='magenta', linewidth=3, arrow_length_ratio=0.2
                    )
                    self.ax.text(
                        p0[0] + N_unit[0]*scale*1.1, p0[1] + N_unit[1]*scale*1.1, p0[2] + N_unit[2]*scale*1.1,
                        f"N {str_N}", color='magenta', fontsize=9, weight='bold'
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

            # Cálculo de Límites Dinámicos
            concat_X = np.concatenate([x.flatten() for x in all_X])
            concat_Y = np.concatenate([y.flatten() for y in all_Y])
            concat_Z = np.concatenate([z.flatten() for z in all_Z])

            max_range = max(
                np.max(np.abs(concat_X)), 
                np.max(np.abs(concat_Y)), 
                np.max(np.abs(concat_Z)),
                3.0
            )
            limit = float(np.ceil(max_range * 1.3))

            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_zlim(-limit, limit)

            # ----------------------------------------------------
            # RENDERIZADO DE EJES COMPLETOS Y NOMBRES EN EXTREMOS
            # ----------------------------------------------------
            grid_steps = np.linspace(-limit, limit, 21)
            gx, gy = np.meshgrid(grid_steps, grid_steps)
            gz = np.zeros_like(gx)
            self.ax.plot_wireframe(gx, gy, gz, color='gray', alpha=0.18, linewidth=0.6)

            self.ax.quiver(-limit, 0, 0, 2*limit, 0, 0, color='red', arrow_length_ratio=0.03, linewidth=2)
            self.ax.quiver(0, -limit, 0, 0, 2*limit, 0, color='green', arrow_length_ratio=0.03, linewidth=2)
            self.ax.quiver(0, 0, 0, 0, 0, limit, color='blue', arrow_length_ratio=0.04, linewidth=2)
            self.ax.plot([0, 0], [0, 0], [0, -limit], color='blue', linestyle='--', linewidth=1.5)

            locator = ticker.MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10])
            ticks = locator.tick_values(-limit, limit)
            ticks = ticks[(ticks >= -limit) & (ticks <= limit) & (np.abs(ticks) > 1e-9)]

            for t in ticks:
                val_str = f"{t:g}"
                
                self.ax.scatter([t], [0], [0], color='red', s=12, zorder=5)
                self.ax.text(t, -limit*0.05, 0, val_str, color='darkred', fontsize=8, ha='center', weight='bold')

                self.ax.scatter([0], [t], [0], color='green', s=12, zorder=5)
                self.ax.text(limit*0.05, t, 0, val_str, color='darkgreen', fontsize=8, ha='center', weight='bold')

                self.ax.scatter([0], [0], [t], color='blue', s=12, zorder=5)
                self.ax.text(-limit*0.05, 0, t, val_str, color='darkblue', fontsize=8, ha='center', weight='bold')

            self.ax.scatter([0], [0], [0], color='black', s=15, zorder=5)

            self.ax.text(limit*1.08, 0, 0, "X", color='red', fontsize=14, weight='bold')
            self.ax.text(0, limit*1.08, 0, "Y", color='green', fontsize=14, weight='bold')
            self.ax.text(0, 0, limit*1.08, "Z", color='blue', fontsize=14, weight='bold')

            self.ax.set_axis_off()

            if elev is not None and azim is not None:
                self.ax.view_init(elev=elev, azim=azim)

            self.canvas.draw()

        except Exception as e:
            self.show_log(f"Error en la entrada o cálculo:\n{str(e)}", is_error=True)

    # ----------------------------------------------------
    # Exportación / Guardado y Carga Completa del Proyecto
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
        """Guarda la totalidad del estado del ejercicio actual."""
        # Asegurarse de que los cambios actuales en la UI queden guardados en la lista de superficies
        if self.surfaces_list and 0 <= self.selected_surface_index < len(self.surfaces_list):
            self.surfaces_list[self.selected_surface_index] = self._get_current_input_data()

        data = {
            "surfaces": self.surfaces_list,
            "selected_index": self.selected_surface_index,
            "u0": self.point_u0.get(),
            "v0": self.point_v0.get(),
            "resolution": self.resolution_entry.get(),
            "vector_format": self.vector_format_seg.get(),
            "switch_normal": bool(self.switch_normal.get()),
            "switch_ru": bool(self.switch_ru.get()),
            "switch_rv": bool(self.switch_rv.get()),
            "switch_tangent": bool(self.switch_tangent.get())
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON File", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                self.show_log("Proyecto completo guardado con éxito.", is_error=False)
            except Exception as e:
                self.show_log(f"Error al guardar proyecto:\n{str(e)}", is_error=True)

    def load_project(self):
        """Restaura absolutamente todo el estado guardado del ejercicio."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON File", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 1. Cargar lista de superficies e índice seleccionado
                if "surfaces" in data and data["surfaces"]:
                    self.surfaces_list = data["surfaces"]
                    self.selected_surface_index = data.get("selected_index", 0)
                    if self.selected_surface_index >= len(self.surfaces_list):
                        self.selected_surface_index = 0
                    
                    self._update_selection_dropdown()
                    # Cargar los textos de la superficie seleccionada
                    active_surf = self.surfaces_list[self.selected_surface_index]
                    self._load_surface_into_entries(active_surf)

                # 2. Cargar Punto u0, v0 y Resolución
                if "u0" in data: 
                    self.point_u0.delete(0, ctk.END); self.point_u0.insert(0, str(data["u0"]))
                if "v0" in data: 
                    self.point_v0.delete(0, ctk.END); self.point_v0.insert(0, str(data["v0"]))
                if "resolution" in data: 
                    self.resolution_entry.delete(0, ctk.END); self.resolution_entry.insert(0, str(data["resolution"]))

                # 3. Cargar Formato de Representación
                if "vector_format" in data:
                    self.vector_format_seg.set(str(data["vector_format"]))

                # 4. Cargar Estados de Conmutadores / Switches
                if "switch_normal" in data:
                    if data["switch_normal"]: self.switch_normal.select()
                    else: self.switch_normal.deselect()

                if "switch_ru" in data:
                    if data["switch_ru"]: self.switch_ru.select()
                    else: self.switch_ru.deselect()

                if "switch_rv" in data:
                    if data["switch_rv"]: self.switch_rv.select()
                    else: self.switch_rv.deselect()

                if "switch_tangent" in data:
                    if data["switch_tangent"]: self.switch_tangent.select()
                    else: self.switch_tangent.deselect()

                # 5. Renderizar
                self.plot_surface()
                self.show_log("Proyecto cargado y sincronizado con éxito.", is_error=False)

            except Exception as e:
                self.show_log(f"Error al abrir archivo:\n{str(e)}", is_error=True)


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ParametricSurfaceApp()
    app.mainloop()