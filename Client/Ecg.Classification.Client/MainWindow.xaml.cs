using System.Windows;
using Ecg.Classification.Client.ViewModels;

namespace Ecg.Classification.Client
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		public MainWindow()
		{
			InitializeComponent();
			DataContext = new MainWindowViewModel();
		}
	}
}
