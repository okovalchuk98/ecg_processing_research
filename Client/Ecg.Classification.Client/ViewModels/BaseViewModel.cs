using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace Ecg.Classification.Client.ViewModels
{
	class BaseViewModel : INotifyPropertyChanged
	{
		public event PropertyChangedEventHandler PropertyChanged;

		protected void OnPropertyChanged([CallerMemberName] string propertyName = "")
		{
			PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
		}

		protected void InvalidateCommands()
		{
			CommandManager.InvalidateRequerySuggested();
		}
	}
}
